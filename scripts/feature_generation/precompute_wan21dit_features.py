#!/usr/bin/env python3
"""Pre-extract Wan2.1 DiT (T2V-14B / T2V-1.3B) intermediate features for
temporal video frames (multi-GPU).

Only **T2V** Wan2.1 checkpoints are supported.  I2V / FLF2V variants need
``clip_fea`` (CLIP image features of the reference frame) and a
conditioning video ``y``; this script only feeds T5 text context, so
``WanModel.forward`` would assert and crash for those variants.  We fail
fast at startup if a non-T2V checkpoint is provided.

This is the Wan2.1 counterpart to ``precompute_wandit_features.py`` (which
targets Wan2.2 TI2V-5B).  Differences from the 2.2 script:

* Loads the **Wan2.1 VAE** (``wan.modules.vae.WanVAE``) instead of
  ``Wan2_2_VAE``.  The VAE checkpoint file is ``Wan2.1_VAE.pth`` and the
  spatial downsampling factor is **8** (vs. 16 for TI2V-5B), while the
  temporal stride remains 4.
* Imports come from ``RoboTwin/policy/pi05/src/videomodel/Wan2.1`` rather than
  ``Wan2.2``.
* DiT API is otherwise identical (``WanModel`` taking
  ``(x_list, t, context, seq_len, ...)``), so the same forward-hook based
  intermediate-layer extraction works unchanged.

Like the 2.2 script, this:
* Runs **one** forward pass of the full DiT at a single noise level (default
  τ ≈ 300 / 1000) and captures the activations of one or more transformer
  blocks via PyTorch forward hooks.
* Supports **per-task T5 text conditioning** (策略 B) by encoding each
  unique LeRobot task description with the umt5-xxl T5 encoder once at
  startup, then re-using the embedding for every sample of that task.
  Use ``--no-per-task-prompt`` to fall back to a shared / zero prompt
  (策略 A).
* Saves features as **per-rank shards** in a tmp dir so that an interrupted
  run can resume by re-launching with the same arguments — already-processed
  sample indices are skipped.

Output (per block):
    ``assets/{config_name}/{repo_id}/wan21dit_cache__{keys}__{frames}__blk{block}_t{timestep}.mmap``
    ``assets/{config_name}/{repo_id}/wan21dit_cache__{keys}__{frames}__blk{block}_t{timestep}.meta.pt``

(The mmap-conversion step is currently commented out at the bottom of
``main`` — same as the 2.2 script — but the per-rank shards are sufficient
for downstream merging via ``merge_wandit_shards.py``.)

Usage (single GPU, single block — 20th layer, T2V-1.3B):
    python scripts/precompute_wan21dit_features.py \\
        --config-name pi05_aloha_video_align \\
        --model-dir src/videomodel/Wan2.1/Wan2.1-T2V-1.3B \\
        --feat-block-indices 19

Usage (single GPU, every 4th block — for 30-block T2V-1.3B → 4,8,12,16,20,24,28):
    python scripts/precompute_wan21dit_features.py \\
        --config-name pi05_aloha_video_align \\
        --model-dir src/videomodel/Wan2.1/Wan2.1-T2V-1.3B \\
        --feat-block-interval 4

Usage (multi-GPU, e.g. 4 GPUs, T2V-14B — 40 blocks, every 5 → 5,10,...,40):
    torchrun --standalone --nproc_per_node=4 scripts/precompute_wan21dit_features.py \\
        --config-name pi05_aloha_video_align \\
        --model-dir src/videomodel/Wan2.1/Wan2.1-T2V-14B \\
        --feat-block-interval 5
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

# ── Wan2.1 default config (matches wan/configs/shared_config.py) ────────────
NUM_TRAIN_TIMESTEPS = 1000
TEXT_DIM = 4096
TEXT_LEN = 512


def _setup_wan_imports():
    """Register lightweight package stubs so relative imports inside wan.modules.* work.

    Same trick as the 2.2 script — avoids executing wan/__init__.py (which
    imports image2video / vace / etc.) while still letting model.py do
    ``from .attention import flash_attention`` and t5.py do
    ``from .tokenizers import HuggingfaceTokenizer``.
    """
    import types

    wan_base = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "src", "videomodel", "Wan2.1",
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

from wan.modules.vae import WanVAE                        # noqa: E402  (Wan2.1 VAE)
from wan.modules.model import WanModel                    # noqa: E402  (needs .attention)
from wan.modules.t5 import T5EncoderModel                 # noqa: E402  (needs .tokenizers)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402


def _prepare_video(frames: np.ndarray, target_hw: int = 224) -> torch.Tensor:
    """Prepare frames [T, C, H, W] -> [1, C, T, H, W] in [-1, 1].

    Pads T so that ``(T - 1) % 4 == 0`` (Wan VAE temporal stride) and
    optionally resizes spatial dims to ``target_hw x target_hw``.
    """
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
        context:   list of [L, C] tensors (one per frame in batch)

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
                        help="Directory containing a Wan2.1 T2V model release "
                             "(e.g. Wan2.1-T2V-1.3B / Wan2.1-T2V-14B). "
                             "Must include Wan2.1_VAE.pth, the DiT diffusers config + "
                             "weights, and (for per-task prompts) "
                             "models_t5_umt5-xxl-enc-bf16.pth + google/umt5-xxl/. "
                             "I2V / FLF2V checkpoints are rejected at startup.")

    blk_group = parser.add_mutually_exclusive_group()
    blk_group.add_argument("--feat-block-indices", type=int, nargs="+", default=None,
                           help="Explicit list of block indices (0-indexed) to extract from")
    blk_group.add_argument("--feat-block-interval", type=int, default=None,
                           help="Extract every N-th block (1-indexed: N, 2N, 3N, ...). "
                                "E.g. --feat-block-interval 4 → blocks 4,8,12,16,... (1-indexed)")

    parser.add_argument("--timestep", type=int, default=300,
                        help="Noise timestep for the single-step forward pass (default: 300)")
    parser.add_argument("--shift", type=float, default=5.0,
                        help="Scheduler shift parameter (default: 5.0, matching Wan2.1 T2V default)")
    parser.add_argument("--target-hw", type=int, default=224,
                        help="Spatial size to resize inputs to before VAE encoding (default: 224). "
                             "Wan2.1 VAE has spatial stride 8, so 224→latent 28→DiT grid 14 (with patch=2).")
    parser.add_argument("--pool-size", type=int, nargs=2, default=[14, 14],
                        help="Spatial pool size for output features (default: 14 14)")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY,
                        help=f"Flush to disk every N samples per rank (default: {SAVE_EVERY})")
    parser.add_argument("--no-per-task-prompt", action="store_true",
                        help="Disable per-task text conditioning. Use a shared empty-string "
                             "prompt for all samples (VEGA-3D style / 策略 A).")
    parser.add_argument("--vae-dtype", choices=["float", "bfloat16", "float16"], default="bfloat16",
                        help="Autocast dtype used inside the Wan2.1 VAE (default: bfloat16). "
                             "Wan2.1's WanVAE itself defaults to float; bf16 is fine and saves memory.")
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

    # ── Load Wan2.1 VAE ──────────────────────────────────────────────────────
    vae_pth = os.path.join(args.model_dir, "Wan2.1_VAE.pth")
    if not os.path.exists(vae_pth):
        raise FileNotFoundError(
            f"Wan2.1 VAE checkpoint not found at {vae_pth}. "
            "Make sure --model-dir points at a Wan2.1 release directory."
        )
    vae_dtype = {
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.vae_dtype]
    vae = WanVAE(vae_pth=vae_pth, device=device, dtype=vae_dtype)
    vae.model.eval()
    if is_main:
        logging.info(f"Loaded Wan2.1 VAE from {vae_pth} (dtype={args.vae_dtype})")

    # ── Load Wan2.1 DiT ──────────────────────────────────────────────────────
    dit_model = WanModel.from_pretrained(args.model_dir).eval().requires_grad_(False).to(device)
    num_blocks = len(dit_model.blocks)
    model_type = getattr(dit_model, "model_type", "t2v")

    # i2v / flf2v variants ``forward()`` asserts ``clip_fea is not None and y is not None``;
    # since we only feed text context, the run would crash deep in the loop. Fail fast here.
    if model_type in ("i2v", "flf2v"):
        raise RuntimeError(
            f"Wan2.1 model_type='{model_type}' requires CLIP image features (clip_fea) "
            f"and a conditioning video (y), neither of which this script provides. "
            f"Use a T2V checkpoint (e.g. Wan2.1-T2V-1.3B / Wan2.1-T2V-14B) instead, "
            f"or extend this script to supply clip_fea + y for i2v / flf2v."
        )

    block_indices = _resolve_block_indices(args.feat_block_indices, args.feat_block_interval, num_blocks)
    if not block_indices:
        raise ValueError(
            f"No blocks selected for extraction. With --feat-block-interval={args.feat_block_interval} "
            f"and num_blocks={num_blocks}, the resolved block list is empty. "
            f"Either lower the interval or pass --feat-block-indices explicitly."
        )
    block_indices_display = [bi + 1 for bi in block_indices]
    if is_main:
        logging.info(
            f"Loaded Wan2.1 DiT from {args.model_dir} | model_type={model_type} "
            f"| {num_blocks} blocks, dim={dit_model.dim}"
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
        td = os.path.join(output_dir, f"_wan21dit_tmp__{keys_tag}__{frames_tag}__blk{bi}_t{args.timestep}")
        os.makedirs(td, exist_ok=True)
        tmp_dirs[bi] = td

    def _out_prefix_for_block(bi: int) -> str:
        return os.path.join(output_dir, f"wan21dit_cache__{keys_tag}__{frames_tag}__blk{bi}_t{args.timestep}")

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
            vid = _prepare_video(frames, target_hw=args.target_hw).to(device)

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
