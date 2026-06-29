#!/usr/bin/env python3
"""Pre-extract DINOv2 features for temporal video frames (multi-GPU).

Run this ONCE before training to avoid DINOv2 inference at every training step.
Supports **incremental checkpointing** — if killed mid-run, simply re-launch
with the same arguments and it will resume from where it left off.

Usage (single GPU):
    python scripts/precompute_dino_features.py \
        --config-name pi05_aloha_video_align \
        --dino-model dinov2_vitb14 \
        --dino-ckpt /path/to/dinov2_vitb14_pretrain.pth

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --standalone --nproc_per_node=4 scripts/precompute_dino_features.py \
        --config-name pi05_aloha_video_align \
        --dino-model dinov2_vitb14

Output is saved to ``assets/{config_name}/{repo_id}/dino_cache__{keys}__{frames}.pt``.
Intermediate per-rank shards are stored in a ``_dino_tmp/`` directory next to the
output file so progress is never lost.

Temporal compression follows the same **4n+1** pattern as Wan2.2 VAE:

- Input frames are padded to 4n+1 by repeating the last frame.
- Frame 0 → 1 temporal feature (independent).
- Every subsequent group of 4 frames → 1 temporal feature (mean-pooled).
- T_in frames → T' = (aligned_T - 1) / 4 + 1 output features.

Examples:  9 frames → T'=3,  16 frames → pad to 17 → T'=5,  17 frames → T'=5.

Cache format::

    {
        sample_idx: {
            cam_name: Tensor[D, T', Hp, Wp]   # float16, spatial layout
            ...
        },
        ...
    }

where T' is the temporally compressed length, Hp = Wp = img_size / patch_size,
D = hidden_dim.  For dinov2_vitb14 @ 224: D=768, Hp=Wp=16.

This mirrors the VAE cache layout [C, T', H', W'] for downstream compatibility.
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SAVE_EVERY = 500

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_dino_model(model_name: str, ckpt_path: str | None, device: torch.device):
    """Load a DINOv2 model.

    Tries local ``dinov2.hub.backbones`` first (for projects that vendor the
    dinov2 package), then falls back to ``torch.hub``.  If *ckpt_path* is
    given the weights are loaded from that file instead of the default.

    Returns (model, patch_size).
    """
    _dinov2_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "dinov2")
    if os.path.isdir(_dinov2_src) and _dinov2_src not in sys.path:
        sys.path.insert(0, _dinov2_src)

    try:
        from dinov2.hub.backbones import __dict__ as _bb
        model = _bb[model_name](pretrained=False)
    except Exception:
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=(ckpt_path is None))

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)

    patch_size = model.patch_size if hasattr(model, "patch_size") else 14
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, patch_size


def _build_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _prepare_frames(frames: np.ndarray, transform: T.Compose) -> torch.Tensor:
    """Convert frames [T, C, H, W] (uint8 or float) -> [T_aligned, 3, img_size, img_size].

    Pads temporally to 4n+1 (same rule as Wan2.2 VAE) by repeating the last frame.
    """
    vid = torch.as_tensor(frames, dtype=torch.float32)
    if vid.max() > 1.5:
        vid = vid / 255.0

    t = vid.shape[0]
    aligned_t = ((t - 1 + 3) // 4) * 4 + 1
    if aligned_t > t:
        pad = aligned_t - t
        vid = torch.cat([vid, vid[-1:].expand(pad, -1, -1, -1)], dim=0)

    return transform(vid)


def _temporal_pool_4n1(feat: torch.Tensor) -> torch.Tensor:
    """4n+1 temporal mean-pooling: [T_aligned, ...] → [T', ...].

    Frame 0 maps to output 0. Every subsequent group of 4 frames is
    mean-pooled into one output feature. Mirrors Wan2.2 VAE temporal
    compression so T' is identical for the same input length.
    """
    first = feat[0:1]
    rest = feat[1:]
    groups = rest.reshape(-1, 4, *rest.shape[1:]).mean(dim=1)
    return torch.cat([first, groups], dim=0)


# ── Shard helpers (identical to VAE script) ─────────────────────────────────

def _shard_path(tmp_dir: str, rank: int, chunk_id: int) -> str:
    return os.path.join(tmp_dir, f"rank{rank}_chunk{chunk_id:06d}.pt")


def _load_existing_shards(tmp_dir: str, rank: int) -> dict:
    cache = {}
    pattern = os.path.join(tmp_dir, f"rank{rank}_chunk*.pt")
    for path in sorted(glob.glob(pattern)):
        cache.update(torch.load(path, map_location="cpu", weights_only=False))
    return cache


def _flush_chunk(tmp_dir: str, rank: int, chunk_id: int, data: dict):
    dst = _shard_path(tmp_dir, rank, chunk_id)
    tmp = dst + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, dst)


def _shards_to_mmap(tmp_dir: str, output_prefix: str):
    """Stream-convert shard .pt files to mmap format without loading all into memory."""
    shard_files = sorted(glob.glob(os.path.join(tmp_dir, "rank*_chunk*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"No shards in {tmp_dir}")

    logging.info(f"Found {len(shard_files)} shard files, scanning keys & shapes ...")
    all_keys = []
    sample_shape = None
    cam_names = None
    for sf in tqdm.tqdm(shard_files, desc="Scanning shards"):
        shard = torch.load(sf, map_location="cpu", weights_only=False)
        for k, v in shard.items():
            all_keys.append(k)
            if sample_shape is None:
                cam_names = list(v.keys())
                sample_shape = v[cam_names[0]].shape
        del shard

    all_keys = sorted(set(all_keys))
    n = len(all_keys)
    n_cams = len(cam_names)
    np_dtype = np.float16
    full_shape = (n, n_cams, *sample_shape)
    logging.info(f"  {n} samples, {n_cams} cameras, shape={sample_shape}")

    mmap_path = output_prefix + ".mmap"
    logging.info(f"Creating mmap {mmap_path}: shape={full_shape}")
    fp = np.memmap(mmap_path, dtype=np_dtype, mode="w+", shape=full_shape)

    key_to_pos = {k: i for i, k in enumerate(all_keys)}
    for sf in tqdm.tqdm(shard_files, desc="Writing to mmap"):
        shard = torch.load(sf, map_location="cpu", weights_only=False)
        for k, per_cam in shard.items():
            pos = key_to_pos[k]
            for j, c in enumerate(cam_names):
                t = per_cam[c]
                if t.dtype == torch.bfloat16:
                    t = t.to(torch.float16)
                fp[pos, j] = t.numpy().astype(np_dtype, copy=False)
        del shard
        fp.flush()
    del fp

    meta = {
        "shape": full_shape,
        "dtype": str(np.dtype(np_dtype)),
        "key_to_pos": key_to_pos,
        "cam_names": cam_names,
    }
    meta_path = output_prefix + ".meta.pt"
    torch.save(meta, meta_path)
    size_gb = os.path.getsize(mmap_path) / 1e9
    logging.info(f"Saved {n} samples → {mmap_path} ({size_gb:.2f} GB) + {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--dino-model", default="dinov2_vitb14",
                        help="DINOv2 model name (default: dinov2_vitb14)")
    parser.add_argument("--dino-ckpt", default=None,
                        help="Path to local DINOv2 checkpoint (.pth). "
                             "If omitted, loads pretrained weights from torch.hub.")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Input image size for DINOv2 (default: 224)")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY,
                        help="Flush to disk every N samples per rank (default: 500)")
    args = parser.parse_args()

    # ── Distributed setup ────────────────────────────────────────────────────
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # ── Config & dataset ─────────────────────────────────────────────────────
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import openpi.training.config as _config

    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.video_delta_frames is None or data_config.video_image_keys is None:
        raise ValueError("Config does not have video_delta_frames / video_image_keys set")

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)

    delta_timestamps = {
        img_key: [f / dataset_meta.fps for f in data_config.video_delta_frames]
        for img_key in data_config.video_image_keys
    }
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, delta_timestamps=delta_timestamps)

    cam_names = [k.split(".")[-1] for k in data_config.video_image_keys]

    # ── Output paths ─────────────────────────────────────────────────────────
    output_dir = str(config.assets_dirs / data_config.repo_id)
    os.makedirs(output_dir, exist_ok=True)
    frames_tag = "f" + "_".join(str(f) for f in data_config.video_delta_frames)
    keys_tag = "+".join(cam_names)
    filename = f"dino_cache__{keys_tag}__{frames_tag}.pt"
    out_path = os.path.join(output_dir, filename)
    tmp_dir = os.path.join(output_dir, f"_dino_tmp__{keys_tag}__{frames_tag}")
    os.makedirs(tmp_dir, exist_ok=True)

    # ── DINOv2 ───────────────────────────────────────────────────────────────
    dino, patch_size = _load_dino_model(args.dino_model, args.dino_ckpt, device)
    transform = _build_transform(args.img_size)
    hp = wp = args.img_size // patch_size
    if is_main:
        logging.info(
            f"Loaded DINOv2 model: {args.dino_model} "
            f"(img_size={args.img_size}, patch={patch_size}, spatial={hp}x{wp})"
        )

    # ── Shard indices across ranks ───────────────────────────────────────────
    all_indices = list(range(len(dataset)))
    shard_indices = all_indices[rank::world_size]

    # ── Resume: load existing progress ───────────────────────────────────────
    existing_cache = _load_existing_shards(tmp_dir, rank)
    done_indices = set(existing_cache.keys())
    todo_indices = [i for i in shard_indices if i not in done_indices]

    if is_main:
        logging.info(
            f"Total {len(dataset)} samples, {world_size} GPUs, ~{len(shard_indices)} per GPU"
        )
    if done_indices:
        logging.info(
            f"[rank {rank}] Resuming: {len(done_indices)} already done, "
            f"{len(todo_indices)} remaining"
        )

    # ── Extract ──────────────────────────────────────────────────────────────
    chunk_id = len(glob.glob(os.path.join(tmp_dir, f"rank{rank}_chunk*.pt")))
    pending_buf: dict = {}
    pbar = tqdm.tqdm(todo_indices, desc=f"[rank {rank}] Extracting", disable=not is_main)
    for idx in pbar:
        sample = dataset[idx]
        per_cam = {}
        for video_key, cam_name in zip(data_config.video_image_keys, cam_names):
            frames = np.asarray(sample[video_key])  # [T, C, H, W]
            imgs = _prepare_frames(frames, transform).to(device)  # [T, 3, H', W']

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                out = dino(imgs, is_training=True)

            patches = out["x_norm_patchtokens"]  # [T_aligned, Hp*Wp, D]
            T_aligned, _, D = patches.shape
            # reshape to spatial: [T_aligned, Hp, Wp, D] → [T_aligned, D, Hp, Wp]
            spatial = (
                patches
                .reshape(T_aligned, hp, wp, D)
                .permute(0, 3, 1, 2)
            )
            # 4n+1 temporal pooling: [T_aligned, D, Hp, Wp] → [T', D, Hp, Wp]
            pooled = _temporal_pool_4n1(spatial)
            # → [D, T', Hp, Wp]  (matches VAE layout [C, T', H', W'])
            feat = pooled.permute(1, 0, 2, 3)

            per_cam[cam_name] = feat.cpu().half()
        pending_buf[idx] = per_cam

        if len(pending_buf) >= args.save_every:
            _flush_chunk(tmp_dir, rank, chunk_id, pending_buf)
            chunk_id += 1
            pending_buf = {}

    if pending_buf:
        _flush_chunk(tmp_dir, rank, chunk_id, pending_buf)

    # ── Barrier — wait for all ranks to finish ───────────────────────────────
    if use_ddp:
        dist.barrier()

    # ── Convert shards directly to mmap on rank 0 (no giant .pt merge) ──────
    if is_main:
        mmap_prefix = os.path.splitext(out_path)[0]
        _shards_to_mmap(tmp_dir, mmap_prefix)

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logging.info(f"Cleaned up temp dir {tmp_dir}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
