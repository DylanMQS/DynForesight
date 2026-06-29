#!/usr/bin/env python3
"""Pre-extract Wan2.2 VAE features for temporal video frames (multi-GPU).

Run this ONCE before training to avoid VAE inference at every training step.
Supports **incremental checkpointing** — if killed mid-run, simply re-launch
with the same arguments and it will resume from where it left off.

Usage (single GPU):
    python scripts/precompute_vae_features.py \
        --config-name pi05_aloha_video_align \
        --vae-dir src/videomodel/Wan2.2/Wan2.2-TI2V-5B

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --standalone --nproc_per_node=4 scripts/precompute_vae_features.py \
        --config-name pi05_aloha_video_align \
        --vae-dir src/videomodel/Wan2.2/Wan2.2-TI2V-5B

Output is saved to ``assets/{config_name}/{repo_id}/vae_cache__{keys}__{frames}.pt``.
Intermediate per-rank shards are stored in a ``_vae_tmp/`` directory next to the
output file so progress is never lost.
"""

import argparse
import glob
import importlib.util
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SAVE_EVERY = 500


def _load_vae_class():
    vae_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "src", "videomodel", "Wan2.2", "wan", "modules", "vae2_2.py",
    )
    spec = importlib.util.spec_from_file_location("vae2_2_standalone", vae_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Wan2_2_VAE


def _prepare_video(frames: np.ndarray, target_hw: int = 224) -> torch.Tensor:
    """Prepare frames [T, C, H, W] -> [1, C, T, H, W] in [-1, 1]."""
    vid = torch.as_tensor(frames, dtype=torch.float32).unsqueeze(0)
    vid = vid.permute(0, 2, 1, 3, 4)

    if vid.max() > 1.5:
        vid = vid / 255.0
    if vid.min() >= 0.0:
        vid = vid * 2.0 - 1.0

    t = vid.shape[2]
    aligned_t = ((t - 1 + 3) // 4) * 4 + 1
    if aligned_t > t:
        pad = aligned_t - t
        vid = torch.cat([vid, vid[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)

    h, w = vid.shape[3], vid.shape[4]
    if h != target_hw or w != target_hw:
        bsz, c, tf, _, _ = vid.shape
        vid = vid.permute(0, 2, 1, 3, 4).reshape(bsz * tf, c, h, w)
        vid = torch.nn.functional.interpolate(vid, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
        vid = vid.reshape(bsz, tf, c, target_hw, target_hw).permute(0, 2, 1, 3, 4)

    return vid


def _shard_path(tmp_dir: str, rank: int, chunk_id: int) -> str:
    return os.path.join(tmp_dir, f"rank{rank}_chunk{chunk_id:06d}.pt")


def _load_existing_shards(tmp_dir: str, rank: int) -> dict:
    """Load all previously saved chunks for this rank and return merged dict."""
    cache = {}
    pattern = os.path.join(tmp_dir, f"rank{rank}_chunk*.pt")
    for path in sorted(glob.glob(pattern)):
        cache.update(torch.load(path, map_location="cpu", weights_only=False))
    return cache


def _load_all_shards(tmp_dir: str) -> dict:
    """Load chunks from ALL ranks in the temp directory."""
    cache = {}
    pattern = os.path.join(tmp_dir, "rank*_chunk*.pt")
    for path in sorted(glob.glob(pattern)):
        cache.update(torch.load(path, map_location="cpu", weights_only=False))
    return cache


def _flush_chunk(tmp_dir: str, rank: int, chunk_id: int, data: dict):
    """Atomically save a chunk to disk."""
    dst = _shard_path(tmp_dir, rank, chunk_id)
    tmp = dst + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--vae-dir", required=True, help="Directory containing Wan2.2_VAE.pth")
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

    # ── Output paths ──────────────────────────────────────────────────────────
    output_dir = str(config.assets_dirs / data_config.repo_id)
    os.makedirs(output_dir, exist_ok=True)
    frames_tag = "f" + "_".join(str(f) for f in data_config.video_delta_frames)
    keys_tag = "+".join(cam_names)
    filename = f"vae_cache__{keys_tag}__{frames_tag}.pt"
    out_path = os.path.join(output_dir, filename)
    tmp_dir = os.path.join(output_dir, f"_vae_tmp__{keys_tag}__{frames_tag}")
    os.makedirs(tmp_dir, exist_ok=True)

    # ── VAE ───────────────────────────────────────────────────────────────────
    Wan2_2_VAE = _load_vae_class()
    vae_pth = os.path.join(args.vae_dir, "Wan2.2_VAE.pth")
    vae = Wan2_2_VAE(vae_pth=vae_pth, device=device, dtype=torch.bfloat16)
    vae.model.eval()
    if is_main:
        logging.info(f"Loaded VAE from {vae_pth}")

    # ── Shard indices across ranks ────────────────────────────────────────────
    all_indices = list(range(len(dataset)))
    shard_indices = all_indices[rank::world_size]

    # ── Resume: load existing progress ────────────────────────────────────────
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

    # ── Extract ───────────────────────────────────────────────────────────────
    chunk_id = len(glob.glob(os.path.join(tmp_dir, f"rank{rank}_chunk*.pt")))
    pending_buf: dict = {}
    pbar = tqdm.tqdm(todo_indices, desc=f"[rank {rank}] Extracting", disable=not is_main)
    for idx in pbar:
        sample = dataset[idx]
        per_cam = {}
        for video_key, cam_name in zip(data_config.video_image_keys, cam_names):
            frames = np.asarray(sample[video_key])
            vid = _prepare_video(frames).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vae.model.clear_cache()
                latent = vae.model.encode(vid, vae.scale)

            per_cam[cam_name] = latent.squeeze(0).cpu().half()
        pending_buf[idx] = per_cam

        if len(pending_buf) >= args.save_every:
            _flush_chunk(tmp_dir, rank, chunk_id, pending_buf)
            chunk_id += 1
            pending_buf = {}

    if pending_buf:
        _flush_chunk(tmp_dir, rank, chunk_id, pending_buf)

    # ── Barrier — wait for all ranks to finish ────────────────────────────────
    if use_ddp:
        dist.barrier()

    # ── Merge all shards on rank 0 ────────────────────────────────────────────
    if is_main:
        cache = _load_all_shards(tmp_dir)

        torch.save(cache, out_path)

        sample_shape = {k: v.shape for k, v in next(iter(cache.values())).items()}
        size_mb = sum(sum(v.nelement() * 2 for v in d.values()) for d in cache.values()) / 1e6
        logging.info(f"Saved {len(cache)} samples x {len(cam_names)} cameras to {out_path}")
        logging.info(f"Per-sample shapes: {sample_shape}, total size: {size_mb:.1f} MB")

        # Clean up temp shards
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logging.info(f"Cleaned up temp dir {tmp_dir}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
