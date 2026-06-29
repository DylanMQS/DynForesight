#!/usr/bin/env python3
"""Pre-extract Wan2.2 DiT (TI2V-5B) intermediate features for temporal video frames (multi-GPU).

Similar to precompute_vae_features.py, but runs a single-step forward pass through
the full DiT at a specific noise level and extracts intermediate-layer features as
visual representations.  Supports extracting from **multiple blocks** in one pass.

Supports **incremental checkpointing** — if killed mid-run, simply re-launch
with the same arguments and it will resume from where it left off.

Text conditioning (策略 B):
    By default, each sample's DiT features are conditioned on its per-task text
    description from the LeRobot dataset (via T5 encoding).  The unique task
    descriptions are encoded once at startup.  Use ``--no-per-task-prompt`` to
    fall back to a shared empty-string prompt (策略 A / VEGA-3D style).

Usage (single GPU, single block — 20th layer):
    python scripts/precompute_wandit_features.py \
        --config-name pi05_aloha_video_align \
        --model-dir src/videomodel/Wan2.2/Wan2.2-TI2V-5B \
        --feat-block-indices 19

Usage (single GPU, multiple blocks at interval 4: layer 4,8,12,16,20,24,28):
    python scripts/precompute_wandit_features.py \
        --config-name pi05_aloha_video_align \
        --model-dir src/videomodel/Wan2.2/Wan2.2-TI2V-5B \
        --feat-block-interval 4

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --standalone --nproc_per_node=4 scripts/precompute_wandit_features.py \
        --config-name pi05_aloha_video_align \
        --model-dir src/videomodel/Wan2.2/Wan2.2-TI2V-5B \
        --feat-block-interval 4

Output is saved per block as memory-mapped files:
    ``assets/{config_name}/{repo_id}/wandit_cache__{keys}__{frames}__blk{block}_t{timestep}.mmap``
    ``assets/{config_name}/{repo_id}/wandit_cache__{keys}__{frames}__blk{block}_t{timestep}.meta.pt``
"""

import argparse
import glob
import logging
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SAVE_EVERY = 500

# ── TI2V-5B default config ──────────────────────────────────────────────────
NUM_TRAIN_TIMESTEPS = 1000
TEXT_DIM = 4096
TEXT_LEN = 512


def _setup_wan_imports():
    """Register lightweight package stubs so relative imports inside wan.modules.* work.

    This avoids executing wan/__init__.py (which pulls in heavy deps like animate,
    speech2video, etc.) while still allowing model.py's ``from .attention import ...``
    and t5.py's ``from .tokenizers import ...`` to resolve correctly.
    """
    import types

    wan_base = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "src", "videomodel", "Wan2.2",
    ))
    wan_dir = os.path.join(wan_base, "wan")

    for pkg_name, pkg_path in [
        ("wan", wan_dir),
        ("wan.modules", os.path.join(wan_dir, "modules")),
        ("wan.utils", os.path.join(wan_dir, "utils")),
    ]:
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [pkg_path]
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg


_setup_wan_imports()

from wan.modules.vae2_2 import Wan2_2_VAE              # noqa: E402
from wan.modules.model import WanModel                  # noqa: E402  (needs .attention)
from wan.modules.t5 import T5EncoderModel               # noqa: E402  (needs .tokenizers)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402


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


def _scan_existing_keys(tmp_dir: str, rank: int) -> set:
    """Return set of sample indices present in existing shards, without keeping tensor data."""
    keys = set()
    pattern = os.path.join(tmp_dir, f"rank{rank}_chunk*.pt")
    for path in sorted(glob.glob(pattern)):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        keys.update(shard.keys())
        del shard
    return keys


def _flush_chunk(tmp_dir: str, rank: int, chunk_id: int, data: dict):
    dst = _shard_path(tmp_dir, rank, chunk_id)
    tmp = dst + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, dst)


def _shards_to_mmap(tmp_dir: str, output_prefix: str):
    """Stream-convert per-block shard .pt files to a single mmap file.

    Shard format: {sample_idx: {cam_name: tensor}}.
    Produces:
        <output_prefix>.mmap      — shape (N, n_cams, *feat_shape)
        <output_prefix>.meta.pt   — {shape, dtype, key_to_pos, cam_names}

    Peak memory ≈ one shard at a time.
    """
    shard_files = sorted(glob.glob(os.path.join(tmp_dir, "rank*_chunk*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"No shards found in {tmp_dir}")

    logging.info(f"Found {len(shard_files)} shard files, scanning keys & shapes ...")
    all_keys: list = []
    sample_shape = None
    cam_names: Optional[List[str]] = None

    for sf in tqdm.tqdm(shard_files, desc="Scanning shards"):
        shard = torch.load(sf, map_location="cpu", weights_only=False)
        for k, v in shard.items():
            all_keys.append(k)
            if sample_shape is None:
                cam_names = sorted(v.keys())
                sample_shape = tuple(v[cam_names[0]].shape)
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
        os.remove(sf)
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


def _setup_scheduler(scheduler_cls, shift: float, device: torch.device):
    scheduler = scheduler_cls(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(NUM_TRAIN_TIMESTEPS, device=device, shift=shift)
    return scheduler


def _select_timestep(timesteps: torch.Tensor, target: int) -> torch.Tensor:
    tau = torch.tensor(target, device=timesteps.device, dtype=timesteps.dtype)
    idx = torch.argmin(torch.abs(timesteps - tau))
    return timesteps[idx]


def _resolve_block_indices(args_indices, args_interval, num_blocks: int) -> List[int]:
    """Determine which block indices to extract from."""
    if args_indices is not None:
        indices = []
        for bi in args_indices:
            resolved = bi if bi >= 0 else num_blocks + bi
            if resolved < 0 or resolved >= num_blocks:
                raise ValueError(f"Block index {bi} out of range [0, {num_blocks})")
            indices.append(resolved)
        return sorted(set(indices))

    if args_interval is not None:
        return [i - 1 for i in range(args_interval, num_blocks + 1, args_interval)]

    return [19]


# ── T5 per-task prompt encoding ──────────────────────────────────────────────

def _encode_task_prompts(
    tasks: Dict[int, str],
    model_dir: str,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """Encode all unique task descriptions with T5, return {task_index: embedding tensor}."""
    t5_ckpt = os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    t5_tok = os.path.join(model_dir, "google", "umt5-xxl")

    if not os.path.exists(t5_ckpt):
        raise FileNotFoundError(
            f"T5 checkpoint not found at {t5_ckpt}. "
            "Per-task prompt encoding requires the T5 weights in the model dir."
        )

    t5 = T5EncoderModel(
        text_len=TEXT_LEN,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=t5_ckpt,
        tokenizer_path=t5_tok,
    )

    unique_texts = sorted(set(tasks.values()))
    logging.info(f"Encoding {len(unique_texts)} unique task descriptions with T5...")

    text_to_emb: Dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for text in unique_texts:
            emb = t5([text], device)[0]  # [seq_len_variable, 4096]
            text_to_emb[text] = emb.detach().cpu()

    task_embs: Dict[int, torch.Tensor] = {}
    for task_idx, task_text in tasks.items():
        task_embs[task_idx] = text_to_emb[task_text]

    del t5
    torch.cuda.empty_cache()
    logging.info(f"T5 encoding done. Freed T5 memory.")

    return task_embs


def _load_fallback_prompt(model_dir: str) -> Optional[torch.Tensor]:
    """Try to load a pre-computed shared prompt embedding file."""
    candidates = [
        os.path.join(model_dir, "wan_prompt_embedding.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "wan_prompt_embedding.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(data, dict):
                data = data.get("context", data.get("embedding", None))
            if torch.is_tensor(data):
                logging.info(f"Loaded shared prompt embedding from {path}")
                return data.detach()
    return None


# ── DiT feature extraction ───────────────────────────────────────────────────

def _extract_dit_features_multi(
    latent_5d: torch.Tensor,
    dit_model,
    scheduler,
    tau: torch.Tensor,
    context: list,
    block_indices: List[int],
    pool_size: tuple,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """Run ONE forward pass, capture features from multiple blocks simultaneously.

    Args:
        latent_5d: VAE latent [1, C_z, T', H', W']
        context: list of [L, C] tensors (one per frame in batch)

    Returns:
        Dict mapping block_idx -> pooled features [T', C_dit, pool_h, pool_w]
    """
    patch_size = dit_model.patch_size

    latent = latent_5d.squeeze(0)  # [C_z, T', H', W']
    c_z, t_lat, h_lat, w_lat = latent.shape

    frame_latents = [latent[:, i:i+1, :, :] for i in range(t_lat)]

    seq_len = math.ceil(
        (h_lat * w_lat) / (patch_size[1] * patch_size[2]) * 1
    )

    latent_batch = torch.stack(frame_latents, dim=0)
    noise = torch.randn_like(latent_batch)
    noisy_batch = scheduler.add_noise(
        original_samples=latent_batch,
        noise=noise,
        timesteps=tau.expand(t_lat),
    )
    noisy_list = [noisy_batch[i] for i in range(t_lat)]
    del latent_batch, noise, noisy_batch

    feat_holders: Dict[int, torch.Tensor] = {}

    def _make_hook(blk_idx):
        def _hook(_, __, output):
            feat_holders[blk_idx] = output.detach()
        return _hook

    handles = []
    for bi in block_indices:
        h = dit_model.blocks[bi].register_forward_hook(_make_hook(bi))
        handles.append(h)

    try:
        t_tensor = tau.expand(t_lat).to(device=device, dtype=torch.long)
        ctx = context * t_lat if len(context) == 1 else context
        _ = dit_model(
            noisy_list,
            t=t_tensor,
            context=ctx,
            seq_len=seq_len,
        )
    finally:
        for h in handles:
            h.remove()

    del noisy_list

    grid_h = h_lat // patch_size[1]
    grid_w = w_lat // patch_size[2]
    tokens_per_frame = grid_h * grid_w

    result = {}
    for bi in block_indices:
        if bi not in feat_holders:
            raise RuntimeError(f"Failed to capture features from block {bi}.")
        feats = feat_holders[bi]
        if feats.shape[1] >= tokens_per_frame:
            feats = feats[:, :tokens_per_frame, :]
        feats = feats.view(feats.shape[0], grid_h, grid_w, feats.shape[2])
        feats = feats.permute(0, 3, 1, 2).contiguous()
        feats = F.adaptive_avg_pool2d(feats, output_size=pool_size)
        result[bi] = feats

    del feat_holders
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing Wan2.2 TI2V-5B model (Wan2.2_VAE.pth + DiT + T5 weights)")

    blk_group = parser.add_mutually_exclusive_group()
    blk_group.add_argument("--feat-block-indices", type=int, nargs="+", default=None,
                           help="Explicit list of block indices (0-indexed) to extract from")
    blk_group.add_argument("--feat-block-interval", type=int, default=None,
                           help="Extract every N-th block (1-indexed: N, 2N, 3N, ...). "
                                "E.g. --feat-block-interval 4 → blocks 4,8,12,16,20,24,28 (1-indexed)")

    parser.add_argument("--timestep", type=int, default=300,
                        help="Noise timestep for the single-step forward pass (default: 300)")
    parser.add_argument("--shift", type=float, default=5.0,
                        help="Scheduler shift parameter (default: 5.0, matching TI2V-5B sample_shift)")
    parser.add_argument("--pool-size", type=int, nargs=2, default=[14, 14],
                        help="Spatial pool size for output features (default: 14 14)")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY,
                        help=f"Flush to disk every N samples per rank (default: {SAVE_EVERY})")
    parser.add_argument("--no-per-task-prompt", action="store_true",
                        help="Disable per-task text conditioning. Use a shared empty-string "
                             "prompt for all samples (VEGA-3D style / 策略 A).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for noise generation reproducibility")
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

    torch.manual_seed(args.seed + rank)

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

    # ── Prepare per-task T5 embeddings (策略 B) or shared prompt (策略 A) ────
    task_embs: Optional[Dict[int, torch.Tensor]] = None
    shared_context: Optional[list] = None

    if not args.no_per_task_prompt and hasattr(dataset_meta, "tasks") and dataset_meta.tasks:
        if is_main:
            logging.info(f"策略 B: encoding {len(dataset_meta.tasks)} task prompts with T5 ...")
        task_embs = _encode_task_prompts(dataset_meta.tasks, args.model_dir, device)
        if is_main:
            sample_task = next(iter(dataset_meta.tasks.values()))
            sample_emb = next(iter(task_embs.values()))
            logging.info(
                f"Per-task prompt ready. Example: \"{sample_task}\" → shape {tuple(sample_emb.shape)}"
            )
    else:
        if is_main:
            logging.info("策略 A: using shared prompt for all samples")
        fallback = _load_fallback_prompt(args.model_dir)
        if fallback is not None:
            shared_context = [fallback.to(device=device, dtype=torch.bfloat16)]
        else:
            if is_main:
                logging.info("No pre-computed prompt embedding found, using zero vector")
            shared_context = [torch.zeros(1, TEXT_DIM, device=device, dtype=torch.bfloat16)]

    # ── Load VAE ─────────────────────────────────────────────────────────────
    vae_pth = os.path.join(args.model_dir, "Wan2.2_VAE.pth")
    vae = Wan2_2_VAE(vae_pth=vae_pth, device=device, dtype=torch.bfloat16)
    vae.model.eval()
    if is_main:
        logging.info(f"Loaded VAE from {vae_pth}")

    # ── Load DiT ─────────────────────────────────────────────────────────────
    dit_model = WanModel.from_pretrained(args.model_dir).eval().requires_grad_(False).to(device)
    num_blocks = len(dit_model.blocks)

    block_indices = _resolve_block_indices(args.feat_block_indices, args.feat_block_interval, num_blocks)
    block_indices_display = [bi + 1 for bi in block_indices]
    if is_main:
        logging.info(
            f"Loaded DiT from {args.model_dir} | {num_blocks} blocks, dim={dit_model.dim}"
        )
        logging.info(
            f"Extracting from blocks (0-indexed): {block_indices}  "
            f"(1-indexed: {block_indices_display})"
        )

    # ── Setup scheduler ──────────────────────────────────────────────────────
    scheduler = _setup_scheduler(FlowUniPCMultistepScheduler, shift=args.shift, device=device)
    tau = _select_timestep(scheduler.timesteps, target=args.timestep)
    if is_main:
        logging.info(f"Scheduler shift={args.shift}, selected timestep tau={tau.item()}")

    # ── Output paths (per-block tmp dirs + per-block output files) ──────────
    output_dir = str(config.assets_dirs / data_config.repo_id)
    os.makedirs(output_dir, exist_ok=True)
    frames_tag = "f" + "_".join(str(f) for f in data_config.video_delta_frames)
    keys_tag = "+".join(cam_names)

    tmp_dirs: Dict[int, str] = {}
    for bi in block_indices:
        td = os.path.join(output_dir, f"_wandit_tmp__{keys_tag}__{frames_tag}__blk{bi}_t{args.timestep}")
        os.makedirs(td, exist_ok=True)
        tmp_dirs[bi] = td

    def _out_prefix_for_block(bi: int) -> str:
        return os.path.join(output_dir, f"wandit_cache__{keys_tag}__{frames_tag}__blk{bi}_t{args.timestep}")

    # ── Shard indices across ranks ───────────────────────────────────────────
    all_indices = list(range(len(dataset)))
    shard_indices = all_indices[rank::world_size]

    # ── Resume: intersect done indices across all per-block tmp dirs ──────────
    per_block_done: List[set] = []
    for bi in block_indices:
        per_block_done.append(_scan_existing_keys(tmp_dirs[bi], rank))
    done_indices = set.intersection(*per_block_done) if per_block_done else set()
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

    pool_size = tuple(args.pool_size)

    # ── Extract (per-block pending buffers for separate shard files) ────────
    chunk_id = len(glob.glob(os.path.join(tmp_dirs[block_indices[0]], f"rank{rank}_chunk*.pt")))
    pending_bufs: Dict[int, dict] = {bi: {} for bi in block_indices}
    pbar = tqdm.tqdm(todo_indices, desc=f"[rank {rank}] Extracting", disable=not is_main)
    for idx in pbar:
        sample = dataset[idx]

        if task_embs is not None:
            task_idx = int(sample["task_index"])
            emb = task_embs[task_idx].to(device=device, dtype=torch.bfloat16)
            sample_context = [emb]
        else:
            sample_context = shared_context

        per_cam_all: Dict[str, Dict[int, torch.Tensor]] = {}
        for video_key, cam_name in zip(data_config.video_image_keys, cam_names):
            frames = np.asarray(sample[video_key])
            vid = _prepare_video(frames).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vae.model.clear_cache()
                latent = vae.model.encode(vid, vae.scale)

                multi_feats = _extract_dit_features_multi(
                    latent_5d=latent,
                    dit_model=dit_model,
                    scheduler=scheduler,
                    tau=tau,
                    context=sample_context,
                    block_indices=block_indices,
                    pool_size=pool_size,
                    device=device,
                )

            per_cam_all[cam_name] = {bi: feat.cpu().half() for bi, feat in multi_feats.items()}

        for bi in block_indices:
            pending_bufs[bi][idx] = {
                cam_name: cam_feats[bi] for cam_name, cam_feats in per_cam_all.items()
            }

        if len(pending_bufs[block_indices[0]]) >= args.save_every:
            for bi in block_indices:
                _flush_chunk(tmp_dirs[bi], rank, chunk_id, pending_bufs[bi])
                pending_bufs[bi] = {}
            chunk_id += 1

    if pending_bufs[block_indices[0]]:
        for bi in block_indices:
            _flush_chunk(tmp_dirs[bi], rank, chunk_id, pending_bufs[bi])

    # ── Barrier — wait for all ranks to finish ───────────────────────────────
    if use_ddp:
        dist.barrier()

    # # ── Convert per-block shards to mmap on rank 0 (streaming, no OOM) ─────
    # if is_main:
    #     import shutil
    #     for bi in block_indices:
    #         logging.info(f"Merging block {bi} shards → mmap ...")
    #         _shards_to_mmap(tmp_dirs[bi], _out_prefix_for_block(bi))
    #         shutil.rmtree(tmp_dirs[bi], ignore_errors=True)
    #         logging.info(f"  Cleaned up {tmp_dirs[bi]}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
