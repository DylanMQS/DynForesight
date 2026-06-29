#!/usr/bin/env python3
"""Pre-extract VJEPA2 features for temporal video frames (multi-GPU).

Run this ONCE before training to avoid VJEPA2 inference at every training step.
Supports **incremental checkpointing** — if killed mid-run, simply re-launch
with the same arguments and it will resume from where it left off.

Usage (single GPU):
    python scripts/precompute_jepa_features.py \
        --config-name pi05_aloha_video_align \
        --jepa-ckpt src/videomodel/vjepa2/checkpoints/vitl.pt

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --standalone --nproc_per_node=4 scripts/precompute_jepa_features.py \
        --config-name pi05_aloha_video_align \
        --jepa-ckpt src/videomodel/vjepa2/checkpoints/vitl.pt

Output is saved to ``assets/{config_name}/{repo_id}/jepa_cache__{keys}__{frames}.pt``.
Intermediate per-rank shards are stored in a ``_jepa_tmp/`` directory next to the
output file so progress is never lost.

Unlike DINO / SigLIP, VJEPA2 is a **video model** — it ingests all frames at once
via 3D tubelet embedding (tubelet_size=2), so there is **no 4n+1 temporal compression**.
Instead, T input frames become T//2 temporal tokens directly.

Cache format::

    {
        sample_idx: {
            cam_name: Tensor[D, T_tokens, Hp, Wp]   # float16, spatial layout
            ...
        },
        ...
    }

where T_tokens = num_frames // tubelet_size, Hp = Wp = crop_size / patch_size,
D = encoder hidden dim.  For ViT-L @ 256 with patch=16, tubelet=2:
D=1024, T_tokens=8, Hp=Wp=16.

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
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SAVE_EVERY = 500

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_jepa_encoder(ckpt_path: str, crop_size: int, num_frames: int,
                       patch_size: int, tubelet_size: int, device: torch.device):
    """Load a VJEPA2 ViT-L encoder.

    Returns (encoder, patch_size, tubelet_size, embed_dim).
    """
    _vjepa2_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "src", "videomodel", "vjepa2",
    )
    if os.path.isdir(_vjepa2_src) and _vjepa2_src not in sys.path:
        sys.path.insert(0, _vjepa2_src)

    from src.models.vision_transformer import vit_large

    encoder = vit_large(
        img_size=(crop_size, crop_size),
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
    )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "encoder" in state_dict:
        state_dict = state_dict["encoder"]
    state_dict = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in state_dict.items()
    }
    encoder.load_state_dict(state_dict, strict=False)

    embed_dim = encoder.embed_dim
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    return encoder, embed_dim


def _build_video_transform(crop_size: int = 256):
    """Build VJEPA2-compatible video transform.

    Expects numpy array [T, H, W, C] uint8, returns tensor [C, T, H, W].
    """
    _vjepa2_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "src", "videomodel", "vjepa2",
    )
    if os.path.isdir(_vjepa2_src) and _vjepa2_src not in sys.path:
        sys.path.insert(0, _vjepa2_src)

    import src.datasets.utils.video.transforms as video_transforms
    import src.datasets.utils.video.volume_transforms as volume_transforms

    short_side_size = int(256.0 / 224 * crop_size)
    return video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(crop_size, crop_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _prepare_video(frames: np.ndarray, transform, tubelet_size: int) -> torch.Tensor:
    """Convert frames [T, C, H, W] → [1, C, T_aligned, crop, crop].

    Pads temporally so T is divisible by tubelet_size (repeating last frame).
    """
    vid = frames
    if vid.dtype != np.uint8:
        if vid.max() <= 1.5:
            vid = (vid * 255).astype(np.uint8)
        else:
            vid = vid.astype(np.uint8)

    # Dataset gives [T, C, H, W] → need [T, H, W, C] for VJEPA2 transforms
    if vid.ndim == 4 and vid.shape[1] in (1, 3):
        vid = np.transpose(vid, (0, 2, 3, 1))

    t = vid.shape[0]
    aligned_t = ((t + tubelet_size - 1) // tubelet_size) * tubelet_size
    if aligned_t > t:
        pad = aligned_t - t
        vid = np.concatenate([vid, np.repeat(vid[-1:], pad, axis=0)], axis=0)

    # transform: [T, H, W, C] numpy → [C, T, H, W] tensor
    tensor = transform(vid)  # [C, T, H, W]
    return tensor.unsqueeze(0)  # [1, C, T, H, W]


# ── Shard helpers (identical to DINO / VAE scripts) ──────────────────────────

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
    parser.add_argument("--jepa-ckpt", required=True,
                        help="Path to VJEPA2 checkpoint (.pt), e.g. checkpoints/vitl.pt")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Input crop size for VJEPA2 (default: 256)")
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Patch size (default: 16)")
    parser.add_argument("--tubelet-size", type=int, default=2,
                        help="Tubelet size for temporal tokenization (default: 2)")
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

    num_frames_raw = len(data_config.video_delta_frames)
    # Align to tubelet_size
    num_frames = ((num_frames_raw + args.tubelet_size - 1) // args.tubelet_size) * args.tubelet_size
    t_tokens = num_frames // args.tubelet_size
    hp = wp = args.crop_size // args.patch_size

    # ── Output paths ─────────────────────────────────────────────────────────
    output_dir = str(config.assets_dirs / data_config.repo_id)
    os.makedirs(output_dir, exist_ok=True)
    frames_tag = "f" + "_".join(str(f) for f in data_config.video_delta_frames)
    keys_tag = "+".join(cam_names)
    filename = f"jepa_cache__{keys_tag}__{frames_tag}.pt"
    out_path = os.path.join(output_dir, filename)
    tmp_dir = os.path.join(output_dir, f"_jepa_tmp__{keys_tag}__{frames_tag}")
    os.makedirs(tmp_dir, exist_ok=True)

    # ── VJEPA2 encoder ───────────────────────────────────────────────────────
    encoder, embed_dim = _load_jepa_encoder(
        args.jepa_ckpt, args.crop_size, num_frames,
        args.patch_size, args.tubelet_size, device,
    )
    transform = _build_video_transform(args.crop_size)
    if is_main:
        logging.info(
            f"Loaded VJEPA2 encoder from {args.jepa_ckpt} "
            f"(crop={args.crop_size}, patch={args.patch_size}, tubelet={args.tubelet_size}, "
            f"frames={num_frames_raw}→{num_frames}, "
            f"tokens=T{t_tokens}×H{hp}×W{wp}, dim={embed_dim})"
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
            vid = _prepare_video(frames, transform, args.tubelet_size).to(device)
            # vid: [1, C, T_aligned, crop, crop]

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                out = encoder(vid)  # [1, T_tokens*Hp*Wp, D]

            patches = out.squeeze(0)  # [T_tokens*Hp*Wp, D]
            # Reshape to spatial-temporal grid: [T_tokens, Hp, Wp, D] → [D, T_tokens, Hp, Wp]
            feat = (
                patches
                .reshape(t_tokens, hp, wp, embed_dim)
                .permute(3, 0, 1, 2)
            )  # [D, T_tokens, Hp, Wp]

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
