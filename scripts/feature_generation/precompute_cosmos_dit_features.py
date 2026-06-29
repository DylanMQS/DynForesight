#!/usr/bin/env python3
"""Pre-extract Cosmos-Predict2.5 DiT intermediate features for temporal video frames (multi-GPU).

Sister script to ``precompute_wandit_features.py``: instead of Wan2.2 TI2V-5B,
this loads the NVIDIA Cosmos-Predict2.5 DiT (``CosmosTransformer3DModel``) via
the HuggingFace ``diffusers`` Cosmos2.5 pipeline and extracts intermediate
block features for visual representation learning.

Pipeline composition (per ``Cosmos2_5_PredictBasePipeline``):
    * Text encoder: ``Qwen2.5-VL`` (Cosmos-Reason1) — emits *concatenated*
      hidden states from every layer (``transformers.Qwen2_5_VLForConditionalGeneration``).
    * Tokenizer / VAE: ``AutoencoderKLWan`` (spatial down 16×, temporal down 4×).
    * DiT: ``CosmosTransformer3DModel`` with ``transformer_blocks`` ModuleList.
    * Scheduler: ``UniPCMultistepScheduler`` (rectified-flow, sigma ∈ [0, 1]).

Text conditioning (策略 B):
    Each sample's DiT features are conditioned on its per-task text description
    from the LeRobot dataset, encoded once at startup with the Qwen2.5-VL
    text encoder.  Use ``--no-per-task-prompt`` to fall back to a shared
    empty-string prompt (策略 A / VEGA-3D style).

Environment requirements (already validated for the pi05 ``.venv``):
    * ``diffusers >= 0.37`` (provides ``Cosmos2_5_PredictBasePipeline`` /
      ``CosmosTransformer3DModel`` / ``AutoencoderKLWan``).
    * ``transformers >= 4.49`` (provides ``Qwen2_5_VLForConditionalGeneration``).
    * ``accelerate >= 0.31`` (used by ``from_pretrained`` for sharded loads).
    * Network access to ``huggingface.co`` (or pre-cached weights under ``HF_HOME``).

Usage (single GPU, default block interval 4):
    python scripts/precompute_cosmos_dit_features.py \
        --config-name pi05_aloha_video_align \
        --feat-block-interval 4

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --standalone --nproc_per_node=4 scripts/precompute_cosmos_dit_features.py \
        --config-name pi05_aloha_video_align \
        --feat-block-interval 4

Output is saved **per block** as per-rank shard files (then optionally merged
into a single mmap), with the checkpoint revision tag baked into the filename
so that pre-trained / post-trained / distilled checkpoints don't collide:
    ``assets/{config_name}/{repo_id}/_cosmosdit_{rev}_tmp__{keys}__{frames}__blk{block}_t{sigma}/``
    ``assets/{config_name}/{repo_id}/cosmosdit_{rev}_cache__{keys}__{frames}__blk{block}_t{sigma}.mmap``
    ``assets/{config_name}/{repo_id}/cosmosdit_{rev}_cache__{keys}__{frames}__blk{block}_t{sigma}.meta.pt``

where ``{rev}`` is ``post`` / ``pre`` / ``distill`` (auto-derived from
``--revision``, override with ``--revision-tag``).
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

# ── Cosmos-Predict2.5 / diffusers defaults ──────────────────────────────────
DEFAULT_MODEL_ID = "nvidia/Cosmos-Predict2.5-2B"
DEFAULT_REVISION = "diffusers/base/post-trained"

# Sigma value (in [0, 1] — rectified-flow scale) at which to query DiT features.
# Roughly matches "timestep 300/1000" in the Wan TI2V scheduler convention.
DEFAULT_SIGMA = 0.3

# Cosmos uses a fixed 512-token prompt context in the diffusers pipeline.
MAX_PROMPT_TOKENS = 512


def _prepare_video(frames: np.ndarray, target_hw: int = 224, temporal_stride: int = 4) -> torch.Tensor:
    """Prepare frames [T, C, H, W] -> [1, C, T, H, W] in [-1, 1].

    Pads time to ``temporal_stride * k + 1`` so the VAE temporal downsampling
    keeps an exact integer number of latent frames.
    """
    vid = torch.as_tensor(frames, dtype=torch.float32).unsqueeze(0)
    vid = vid.permute(0, 2, 1, 3, 4)

    if vid.max() > 1.5:
        vid = vid / 255.0
    if vid.min() >= 0.0:
        vid = vid * 2.0 - 1.0

    t = vid.shape[2]
    aligned_t = ((t - 1 + (temporal_stride - 1)) // temporal_stride) * temporal_stride + 1
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


def _auto_revision_tag(revision: str) -> str:
    """Derive a short filesystem-safe tag from a HuggingFace revision string.

    ``diffusers/base/post-trained`` -> ``post``
    ``diffusers/base/pre-trained``  -> ``pre``
    ``diffusers/base/distilled``    -> ``distill``
    Anything else -> last path component, sanitized.
    """
    last = revision.rstrip("/").split("/")[-1]
    last = last.replace("-trained", "")
    last = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in last)
    return last or "rev"


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

    return [num_blocks // 2 - 1]  # middle block by default


# ── Qwen2.5-VL per-task prompt encoding (策略 B) ─────────────────────────────

def _encode_prompt_with_qwen(
    text_encoder,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_sequence_length: int = MAX_PROMPT_TOKENS,
) -> torch.Tensor:
    """Replicates ``Cosmos2_5_PredictBasePipeline._get_prompt_embeds``:

    Wraps each prompt in the system+user chat template, tokenizes (pad-to-max),
    runs Qwen2.5-VL in inference, and concatenates *all* normalized hidden
    states (excluding the embeddings layer) along the channel dim.

    Returns a tensor of shape ``[B, max_sequence_length, n_layers * hidden_dim]``.
    """
    input_ids_batch = []
    for p in prompts:
        conversations = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant who will provide prompts to an image generator.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": p}],
            },
        ]
        ids = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            add_vision_id=False,
            max_length=max_sequence_length,
            truncation=True,
            padding="max_length",
        )
        if not isinstance(ids, list):
            ids = ids["input_ids"] if "input_ids" in ids else ids
        input_ids_batch.append(torch.as_tensor(ids, dtype=torch.long))

    input_ids_batch = torch.stack(input_ids_batch, dim=0).to(device)

    with torch.inference_mode():
        outputs = text_encoder(input_ids_batch, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple len = n_layers + 1
    normalized = []
    for layer_idx in range(1, len(hidden_states)):
        hs = hidden_states[layer_idx]
        normed = (hs - hs.mean(dim=-1, keepdim=True)) / (hs.std(dim=-1, keepdim=True) + 1e-8)
        normalized.append(normed)
    prompt_embeds = torch.cat(normalized, dim=-1)
    return prompt_embeds


def _encode_task_prompts(
    tasks: Dict[int, str],
    text_encoder,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, torch.Tensor]:
    """Encode all unique task descriptions with Qwen2.5-VL, return {task_index: embedding tensor (CPU)}."""
    unique_texts = sorted(set(tasks.values()))
    logging.info(f"Encoding {len(unique_texts)} unique task descriptions with Qwen2.5-VL...")

    text_to_emb: Dict[str, torch.Tensor] = {}
    # Encode one-by-one to keep peak memory low (sequence is 512 tokens × ~28 layers × ~3.5k dim).
    for text in tqdm.tqdm(unique_texts, desc="Encoding prompts"):
        emb = _encode_prompt_with_qwen(text_encoder, tokenizer, [text], device=device)
        text_to_emb[text] = emb[0].detach().to(dtype=dtype, device="cpu")

    task_embs: Dict[int, torch.Tensor] = {}
    for task_idx, task_text in tasks.items():
        task_embs[task_idx] = text_to_emb[task_text]
    return task_embs


# ── DiT feature extraction ───────────────────────────────────────────────────

def _add_rectified_flow_noise(clean: torch.Tensor, sigma: float, generator=None) -> torch.Tensor:
    """Rectified-flow forward process: x_t = (1 - σ) x_0 + σ ε,  ε ~ N(0, I).

    Noise is sampled in float32 (some PyTorch versions don't support
    ``normal_(generator=...)`` directly on bfloat16) and cast back.
    """
    noise = torch.empty(clean.shape, dtype=torch.float32, device=clean.device)
    if generator is None:
        noise.normal_()
    else:
        noise.normal_(generator=generator)
    noise = noise.to(dtype=clean.dtype)
    return (1.0 - sigma) * clean + sigma * noise


def _extract_dit_features_multi(
    clean_latent_5d: torch.Tensor,
    transformer,
    sigma: float,
    prompt_embeds: torch.Tensor,
    block_indices: List[int],
    pool_size: tuple,
    device: torch.device,
    dtype: torch.dtype,
    pixel_resolution: int,
    generator: Optional[torch.Generator] = None,
) -> Dict[int, torch.Tensor]:
    """Run ONE forward pass through the Cosmos DiT, capture features from multiple blocks.

    Args:
        clean_latent_5d:  VAE-encoded *normalized* latent ``[1, C_z, T_lat, H_lat, W_lat]``.
        prompt_embeds:    ``[1, L, D_text_concat]`` from ``_encode_prompt_with_qwen``.
        pixel_resolution: Original (square) pixel-side length, only used to size the
                          all-zero ``padding_mask`` — the transformer NEAREST-resizes
                          this internally so the exact size is non-functional, but
                          we keep it physically correct for clarity.

    Returns:
        Dict mapping block_idx -> pooled features ``[T_post, C_dit, pool_h, pool_w]``.
    """
    B, C_z, T_lat, H_lat, W_lat = clean_latent_5d.shape
    assert B == 1, "feature extraction expects batch size 1"

    noisy_latent = _add_rectified_flow_noise(
        clean_latent_5d.to(dtype=dtype), sigma=sigma, generator=generator
    )

    cond_mask = torch.zeros((B, 1, T_lat, H_lat, W_lat), dtype=dtype, device=device)
    padding_mask = torch.zeros((1, 1, pixel_resolution, pixel_resolution), dtype=dtype, device=device)
    timestep = torch.full((B,), float(sigma), dtype=dtype, device=device)

    feat_holders: Dict[int, torch.Tensor] = {}

    def _make_hook(blk_idx):
        def _hook(_mod, _inp, output):
            feat_holders[blk_idx] = output.detach()
        return _hook

    handles = []
    for bi in block_indices:
        h = transformer.transformer_blocks[bi].register_forward_hook(_make_hook(bi))
        handles.append(h)

    try:
        with torch.inference_mode():
            _ = transformer(
                hidden_states=noisy_latent,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds.to(device=device, dtype=dtype),
                condition_mask=cond_mask,
                padding_mask=padding_mask,
                return_dict=False,
            )
    finally:
        for h in handles:
            h.remove()

    p_t, p_h, p_w = transformer.config.patch_size
    post_T = T_lat // p_t
    post_H = H_lat // p_h
    post_W = W_lat // p_w
    expected_tokens = post_T * post_H * post_W

    result: Dict[int, torch.Tensor] = {}
    for bi in block_indices:
        if bi not in feat_holders:
            raise RuntimeError(f"Failed to capture features from block {bi}.")
        feats = feat_holders[bi]  # [B, THW, D]
        if feats.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Block {bi} returned {feats.shape[1]} tokens but expected {expected_tokens} "
                f"({post_T}×{post_H}×{post_W})."
            )
        D = feats.shape[2]
        feats = feats.view(B, post_T, post_H, post_W, D)
        feats = feats.squeeze(0)                     # [T, H, W, D]
        feats = feats.permute(0, 3, 1, 2).contiguous()  # [T, D, H, W]
        feats = F.adaptive_avg_pool2d(feats, output_size=pool_size)
        result[bi] = feats

    del feat_holders
    return result


def _vae_encode_to_latent(vae, video_5d: torch.Tensor) -> torch.Tensor:
    """VAE encode → normalized latent in DiT input space."""
    with torch.inference_mode():
        out = vae.encode(video_5d.to(dtype=vae.dtype))
        if hasattr(out, "latent_dist"):
            latent = out.latent_dist.mode()
        elif hasattr(out, "latents"):
            latent = out.latents
        else:
            raise AttributeError("VAE encode output has no latent_dist or latents")
    latents_mean = torch.as_tensor(vae.config.latents_mean, device=latent.device, dtype=latent.dtype)
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = torch.as_tensor(vae.config.latents_std, device=latent.device, dtype=latent.dtype)
    latents_std = latents_std.view(1, vae.config.z_dim, 1, 1, 1)
    return (latent - latents_mean) / latents_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help=f"HuggingFace model id (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help=f"HF revision/branch for the diffusers checkpoint (default: {DEFAULT_REVISION})")
    parser.add_argument("--revision-tag", default=None,
                        help="Short tag baked into output filenames to disambiguate checkpoints "
                             "(default: auto-derived from --revision, e.g. 'post' / 'pre' / 'distill').")
    parser.add_argument("--transformer-path", default=None,
                        help="Optional local path to a diffusers-format transformer directory "
                             "(e.g. from convert_cosmos_robot_to_diffusers.py). When set, the "
                             "transformer is loaded from this path instead of from --model-id. "
                             "VAE and text encoder are still loaded from --model-id.")

    blk_group = parser.add_mutually_exclusive_group()
    blk_group.add_argument("--feat-block-indices", type=int, nargs="+", default=None,
                           help="Explicit list of block indices (0-indexed) to extract from")
    blk_group.add_argument("--feat-block-interval", type=int, default=None,
                           help="Extract every N-th block (1-indexed: N, 2N, 3N, ...). "
                                "E.g. --feat-block-interval 4 → blocks 4,8,12,16,20,24,28 (1-indexed)")

    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help=f"Noise level σ ∈ [0, 1] for the single-step forward pass "
                             f"(rectified-flow scale; default: {DEFAULT_SIGMA})")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Spatial resolution to resize input frames to (square, default: 224)")
    parser.add_argument("--pool-size", type=int, nargs=2, default=[14, 14],
                        help="Spatial pool size for output features (default: 14 14)")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY,
                        help=f"Flush to disk every N samples per rank (default: {SAVE_EVERY})")
    parser.add_argument("--no-per-task-prompt", action="store_true",
                        help="Disable per-task text conditioning. Use a shared empty-string "
                             "prompt for all samples (VEGA-3D style / 策略 A).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for noise generation reproducibility")
    parser.add_argument("--cache-dir", default=None,
                        help="Override HuggingFace cache dir (otherwise uses HF_HOME/HF_HUB_CACHE)")
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

    # ── Load Cosmos diffusers components ─────────────────────────────────────
    # We deliberately avoid loading ``Cosmos2_5_PredictBasePipeline`` directly to
    # sidestep the ``cosmos_guardrail`` SafetyChecker dependency: we only need
    # text encoder + VAE + transformer for feature extraction, never inference.
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    from diffusers import AutoencoderKLWan, CosmosTransformer3DModel

    common_kwargs = dict(revision=args.revision, torch_dtype=torch.bfloat16)
    if args.cache_dir is not None:
        common_kwargs["cache_dir"] = args.cache_dir

    if is_main:
        logging.info(f"Loading Cosmos diffusers components from {args.model_id} (revision={args.revision}) ...")

    if args.transformer_path is not None:
        if is_main:
            logging.info(f"Loading transformer OVERRIDE from local path: {args.transformer_path}")
        transformer = CosmosTransformer3DModel.from_pretrained(
            args.transformer_path, torch_dtype=torch.bfloat16
        ).eval().to(device)
    else:
        transformer = CosmosTransformer3DModel.from_pretrained(
            args.model_id, subfolder="transformer", **common_kwargs
        ).eval().to(device)
    transformer.requires_grad_(False)

    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", **common_kwargs
    ).eval().to(device)
    vae.requires_grad_(False)

    num_blocks = len(transformer.transformer_blocks)
    block_indices = _resolve_block_indices(args.feat_block_indices, args.feat_block_interval, num_blocks)
    block_indices_display = [bi + 1 for bi in block_indices]
    if is_main:
        hidden_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        logging.info(
            f"Loaded CosmosTransformer3DModel | {num_blocks} blocks, hidden_dim={hidden_dim}, "
            f"patch_size={transformer.config.patch_size}"
        )
        logging.info(
            f"Extracting from blocks (0-indexed): {block_indices}  "
            f"(1-indexed: {block_indices_display})"
        )

    # ── Prepare per-task prompt embeddings (策略 B) or shared prompt (策略 A) ──
    task_embs: Optional[Dict[int, torch.Tensor]] = None
    shared_prompt_embeds: Optional[torch.Tensor] = None

    use_per_task = (
        not args.no_per_task_prompt
        and hasattr(dataset_meta, "tasks")
        and dataset_meta.tasks
    )

    if is_main:
        logging.info(f"Loading Qwen2.5-VL text encoder from {args.model_id} ...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, subfolder="text_encoder", **common_kwargs
    ).eval().to(device)
    text_encoder.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer", revision=args.revision,
        cache_dir=args.cache_dir if args.cache_dir is not None else None,
    )

    if use_per_task:
        if is_main:
            logging.info(f"策略 B: encoding {len(dataset_meta.tasks)} task prompts with Qwen2.5-VL ...")
        task_embs = _encode_task_prompts(
            dataset_meta.tasks, text_encoder, tokenizer, device=device, dtype=torch.bfloat16
        )
        if is_main:
            sample_task = next(iter(dataset_meta.tasks.values()))
            sample_emb = next(iter(task_embs.values()))
            logging.info(
                f"Per-task prompt ready. Example: \"{sample_task}\" → shape {tuple(sample_emb.shape)}"
            )
    else:
        if is_main:
            logging.info("策略 A: encoding a single shared empty prompt for all samples")
        shared_prompt_embeds = _encode_prompt_with_qwen(
            text_encoder, tokenizer, [""], device=device
        ).detach().to(dtype=torch.bfloat16, device="cpu")

    # Free Qwen2.5-VL — it's the heaviest component and unused after encoding.
    del text_encoder, tokenizer
    torch.cuda.empty_cache()
    if is_main:
        logging.info("Freed text encoder; proceeding with VAE + DiT only.")

    # ── Output paths (per-block tmp dirs + per-block output files) ──────────
    output_dir = str(config.assets_dirs / data_config.repo_id)
    os.makedirs(output_dir, exist_ok=True)
    frames_tag = "f" + "_".join(str(f) for f in data_config.video_delta_frames)
    keys_tag = "+".join(cam_names)
    tau_tag = f"{int(round(args.sigma * 1000)):04d}"  # e.g. sigma=0.3 -> "0300"
    rev_tag = args.revision_tag or _auto_revision_tag(args.revision)
    if is_main:
        logging.info(f"Revision tag for output filenames: '{rev_tag}' (from revision='{args.revision}')")

    tmp_dirs: Dict[int, str] = {}
    for bi in block_indices:
        td = os.path.join(
            output_dir,
            f"_cosmosdit_{rev_tag}_tmp__{keys_tag}__{frames_tag}__blk{bi}_t{tau_tag}",
        )
        os.makedirs(td, exist_ok=True)
        tmp_dirs[bi] = td

    def _out_prefix_for_block(bi: int) -> str:
        return os.path.join(
            output_dir,
            f"cosmosdit_{rev_tag}_cache__{keys_tag}__{frames_tag}__blk{bi}_t{tau_tag}",
        )

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

    # Per-rank deterministic noise generator
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + rank)

    # ── Extract (per-block pending buffers for separate shard files) ────────
    chunk_id = len(glob.glob(os.path.join(tmp_dirs[block_indices[0]], f"rank{rank}_chunk*.pt")))
    pending_bufs: Dict[int, dict] = {bi: {} for bi in block_indices}
    pbar = tqdm.tqdm(todo_indices, desc=f"[rank {rank}] Extracting", disable=not is_main)
    for idx in pbar:
        sample = dataset[idx]

        if task_embs is not None:
            task_idx = int(sample["task_index"])
            sample_prompt_embeds = task_embs[task_idx].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        else:
            sample_prompt_embeds = shared_prompt_embeds.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

        per_cam_all: Dict[str, Dict[int, torch.Tensor]] = {}
        for video_key, cam_name in zip(data_config.video_image_keys, cam_names):
            frames = np.asarray(sample[video_key])
            vid = _prepare_video(frames, target_hw=args.resolution).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clean_latent = _vae_encode_to_latent(vae, vid)

                multi_feats = _extract_dit_features_multi(
                    clean_latent_5d=clean_latent,
                    transformer=transformer,
                    sigma=args.sigma,
                    prompt_embeds=sample_prompt_embeds,
                    block_indices=block_indices,
                    pool_size=pool_size,
                    device=device,
                    dtype=torch.bfloat16,
                    pixel_resolution=args.resolution,
                    generator=noise_gen,
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
