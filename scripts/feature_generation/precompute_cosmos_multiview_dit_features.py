#!/usr/bin/env python3
"""Pre-extract Cosmos-Predict2.5 robot/multiview-agibot DiT intermediate features (multi-GPU).

Variant of ``precompute_cosmos_dit_features.py`` that uses the NVIDIA *native*
Cosmos-Predict2.5 checkpoint (``robot/multiview-agibot``) instead of the HuggingFace
diffusers pipeline.  The native model class is ``CameraMiniTrainDITwithConditionalMask``
(28 blocks, 2048 hidden_dim for the 2B variant) loaded via ``Video2WorldInference``.

Key architectural differences from the diffusers-based script:
    * DiT blocks live at ``model.net.blocks`` (nn.ModuleList), not
      ``transformer.transformer_blocks``.
    * Block output shape is ``[B, T, H, W, D]`` (not ``[B, THW, D]``).
    * VAE encoding is done via ``model.encode(video)`` which wraps the
      internal tokenizer and normalization.
    * Text embeddings come from the model's built-in Qwen2.5-VL text encoder
      via ``model.text_encoder.compute_text_embeddings_online()``.
    * Camera conditioning (Plücker rays) can be provided but defaults to
      *disabled* (zeros) for pure visual-feature extraction.

Environment requirements:
    * The ``cosmos-predict2.5`` repo must be cloned and importable (its path
      is auto-added to ``sys.path``).
    * ``megatron-core``, ``transformer_engine >= 2.0``, ``einops``, ``peft``
      and the usual PyTorch + CUDA stack.
    * The robot/multiview-agibot checkpoint must be downloaded (the script
      uses the NVIDIA checkpoint DB to resolve the S3 URI automatically).

Usage (single GPU):
    python scripts/precompute_cosmos_multiview_dit_features.py \\
        --config-name pi05_aloha_video_align \\
        --feat-block-interval 4

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --standalone --nproc_per_node=4 \\
        scripts/precompute_cosmos_multiview_dit_features.py \\
        --config-name pi05_aloha_video_align \\
        --feat-block-interval 4

Output layout mirrors the diffusers-based script, with ``cosmosmv`` prefix:
    ``assets/{config_name}/{repo_id}/cosmosmv_cache__{keys}__{frames}__blk{block}_t{sigma}.mmap``
"""

import argparse
import glob
import logging
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

# Default noise level (rectified-flow σ ∈ [0, 1]).
DEFAULT_SIGMA = 0.3

# ── Path to the cloned cosmos-predict2.5 repo ────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COSMOS_REPO = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "src", "videomodel", "cosmos-predict2.5"))


def _prepare_video(frames: np.ndarray, target_hw: int = 224, temporal_stride: int = 4) -> torch.Tensor:
    """Prepare frames [T, C, H, W] -> [1, C, T, H, W] in [-1, 1]."""
    vid = torch.as_tensor(frames, dtype=torch.float32).unsqueeze(0)
    vid = vid.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

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
        vid = F.interpolate(vid, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
        vid = vid.reshape(bsz, tf, c, target_hw, target_hw).permute(0, 2, 1, 3, 4)

    return vid


def _shard_path(tmp_dir: str, rank: int, chunk_id: int) -> str:
    return os.path.join(tmp_dir, f"rank{rank}_chunk{chunk_id:06d}.pt")


def _scan_existing_keys(tmp_dir: str, rank: int) -> set:
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
    """Stream-convert per-block shard .pt files to a single mmap file."""
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
    logging.info(f"Saved {n} samples -> {mmap_path} ({size_gb:.2f} GB) + {meta_path}")


def _resolve_block_indices(args_indices, args_interval, num_blocks: int) -> List[int]:
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

    return [num_blocks // 2 - 1]


# ── Rectified-flow noise ─────────────────────────────────────────────────────

def _add_rectified_flow_noise(clean: torch.Tensor, sigma: float, generator=None) -> torch.Tensor:
    noise = torch.empty(clean.shape, dtype=torch.float32, device=clean.device)
    if generator is None:
        noise.normal_()
    else:
        noise.normal_(generator=generator)
    noise = noise.to(dtype=clean.dtype)
    return (1.0 - sigma) * clean + sigma * noise


# ── DiT feature extraction (native Cosmos model) ─────────────────────────────

def _extract_dit_features_multi(
    clean_latent_5d: torch.Tensor,
    cosmos_model,
    sigma: float,
    text_embeddings: torch.Tensor,
    block_indices: List[int],
    pool_size: tuple,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> Dict[int, torch.Tensor]:
    """Run ONE forward pass through the native Cosmos DiT and capture block features.

    The model.net is a ``CameraMiniTrainDITwithConditionalMask`` which expects:
        x_B_C_T_H_W, timesteps_B_T, crossattn_emb,
        condition_video_input_mask_B_C_T_H_W, padding_mask, camera, ...

    Block outputs have shape ``[B, T_post, H_post, W_post, D]``.
    """
    net = cosmos_model.net

    B, C_z, T_lat, H_lat, W_lat = clean_latent_5d.shape
    assert B == 1

    noisy_latent = _add_rectified_flow_noise(
        clean_latent_5d.to(dtype=dtype), sigma=sigma, generator=generator
    )

    condition_mask = torch.zeros((B, 1, T_lat, H_lat, W_lat), dtype=dtype, device=device)

    pixel_h = H_lat * cosmos_model.tokenizer.spatial_compression_factor
    pixel_w = W_lat * cosmos_model.tokenizer.spatial_compression_factor
    padding_mask = torch.zeros((B, 1, pixel_h, pixel_w), dtype=dtype, device=device)

    timesteps = torch.full((B, 1), float(sigma), dtype=dtype, device=device)

    feat_holders: Dict[int, torch.Tensor] = {}

    def _make_hook(blk_idx):
        def _hook(_mod, _inp, output):
            feat_holders[blk_idx] = output.detach()
        return _hook

    handles = []
    for bi in block_indices:
        h = net.blocks[bi].register_forward_hook(_make_hook(bi))
        handles.append(h)

    try:
        with torch.inference_mode():
            _ = net(
                x_B_C_T_H_W=noisy_latent,
                timesteps_B_T=timesteps,
                crossattn_emb=text_embeddings.to(device=device, dtype=dtype),
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                padding_mask=padding_mask,
                camera=None,
            )
    finally:
        for h in handles:
            h.remove()

    p_s = net.patch_spatial
    p_t = net.patch_temporal
    post_T = T_lat // p_t
    post_H = H_lat // p_s
    post_W = W_lat // p_s

    result: Dict[int, torch.Tensor] = {}
    for bi in block_indices:
        if bi not in feat_holders:
            raise RuntimeError(f"Failed to capture features from block {bi}.")
        feats = feat_holders[bi]  # [B, T, H, W, D]
        if feats.ndim == 5:
            feats = feats.squeeze(0)              # [T, H, W, D]
            feats = feats.permute(0, 3, 1, 2)     # [T, D, H, W]
        elif feats.ndim == 3:
            D = feats.shape[2]
            feats = feats.view(B, post_T, post_H, post_W, D)
            feats = feats.squeeze(0)
            feats = feats.permute(0, 3, 1, 2)
        else:
            raise RuntimeError(f"Unexpected feature shape from block {bi}: {feats.shape}")

        feats = F.adaptive_avg_pool2d(feats, output_size=pool_size)
        result[bi] = feats

    del feat_holders
    return result


# ── Text encoding via model's built-in Qwen2.5-VL ───────────────────────────

def _encode_task_prompts_native(
    tasks: Dict[int, str],
    cosmos_model,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, torch.Tensor]:
    """Encode task descriptions using the model's built-in text encoder."""
    text_encoder = cosmos_model.text_encoder
    if text_encoder is None:
        logging.warning("Model has no text_encoder; using zero embeddings.")
        return {}

    unique_texts = sorted(set(tasks.values()))
    logging.info(f"Encoding {len(unique_texts)} unique task descriptions with model text encoder...")

    text_to_emb: Dict[str, torch.Tensor] = {}
    for text in tqdm.tqdm(unique_texts, desc="Encoding prompts"):
        emb = text_encoder.compute_text_embeddings_online(
            data_batch={"ai_caption": [text], "images": None},
            input_caption_key="ai_caption",
        )
        text_to_emb[text] = emb[0].detach().to(dtype=dtype, device="cpu")

    task_embs: Dict[int, torch.Tensor] = {}
    for task_idx, task_text in tasks.items():
        task_embs[task_idx] = text_to_emb[task_text]
    return task_embs


def _encode_empty_prompt_native(
    cosmos_model,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode an empty prompt via the model's text encoder."""
    text_encoder = cosmos_model.text_encoder
    if text_encoder is None:
        logging.warning("Model has no text_encoder; returning zero embedding placeholder.")
        return torch.zeros(1, 512, 3584, dtype=dtype)

    emb = text_encoder.compute_text_embeddings_online(
        data_batch={"ai_caption": [""], "images": None},
        input_caption_key="ai_caption",
    )
    return emb.detach().to(dtype=dtype, device="cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract DiT features from Cosmos robot/multiview-agibot model (native NVIDIA loading)."
    )
    parser.add_argument("--config-name", required=True)

    parser.add_argument("--cosmos-repo", default=_COSMOS_REPO,
                        help=f"Path to the cloned cosmos-predict2.5 repo (default: {_COSMOS_REPO})")
    parser.add_argument("--experiment-name", default=None,
                        help="NVIDIA experiment name for the robot multiview model. "
                             "Auto-resolved from checkpoint DB if not provided.")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Local or S3 path to the robot/multiview-agibot checkpoint. "
                             "Auto-resolved from the model registry if not provided.")
    parser.add_argument("--config-file", default=None,
                        help="Config file for the multiview camera model. "
                             "Defaults to the standard multiview_camera config.")

    blk_group = parser.add_mutually_exclusive_group()
    blk_group.add_argument("--feat-block-indices", type=int, nargs="+", default=None)
    blk_group.add_argument("--feat-block-interval", type=int, default=None)

    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--pool-size", type=int, nargs=2, default=[14, 14])
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--no-per-task-prompt", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-tag", default="cosmosmv",
                        help="Prefix tag for output filenames (default: cosmosmv)")
    args = parser.parse_args()

    # ── Add cosmos-predict2.5 to path ─────────────────────────────────────────
    cosmos_repo = os.path.abspath(args.cosmos_repo)
    if not os.path.isdir(cosmos_repo):
        raise FileNotFoundError(f"cosmos-predict2.5 repo not found at {cosmos_repo}")
    if cosmos_repo not in sys.path:
        sys.path.insert(0, cosmos_repo)

    # ── Distributed setup ─────────────────────────────────────────────────────
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

    # ── Config & dataset ──────────────────────────────────────────────────────
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

    # ── Load Cosmos robot/multiview-agibot via native pipeline ────────────────
    from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey, ModelVariant

    model_key = ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT)
    checkpoint_info = MODEL_CHECKPOINTS[model_key]

    experiment_name = args.experiment_name or checkpoint_info.experiment
    checkpoint_path = args.checkpoint_path or checkpoint_info.s3.uri
    config_file = args.config_file or "cosmos_predict2/_src/predict2/camera/configs/multiview_camera/config.py"

    if is_main:
        logging.info(f"Loading Cosmos robot/multiview-agibot model ...")
        logging.info(f"  experiment: {experiment_name}")
        logging.info(f"  checkpoint: {checkpoint_path}")
        logging.info(f"  config_file: {config_file}")

    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    vid2world = Video2WorldInference(
        experiment_name=experiment_name,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=config_file,
    )
    cosmos_model = vid2world.model
    cosmos_model.eval()

    net = cosmos_model.net
    num_blocks = len(net.blocks)
    block_indices = _resolve_block_indices(args.feat_block_indices, args.feat_block_interval, num_blocks)

    if is_main:
        hidden_dim = net.model_channels
        logging.info(
            f"Loaded DiT: {type(net).__name__} | {num_blocks} blocks, "
            f"hidden_dim={hidden_dim}, patch_spatial={net.patch_spatial}, patch_temporal={net.patch_temporal}"
        )
        logging.info(f"Extracting from blocks (0-indexed): {block_indices}")

    # ── Per-task prompt embeddings ────────────────────────────────────────────
    task_embs: Optional[Dict[int, torch.Tensor]] = None
    shared_prompt_embeds: Optional[torch.Tensor] = None

    use_per_task = (
        not args.no_per_task_prompt
        and hasattr(dataset_meta, "tasks")
        and dataset_meta.tasks
    )

    if use_per_task:
        if is_main:
            logging.info(f"Encoding {len(dataset_meta.tasks)} task prompts ...")
        task_embs = _encode_task_prompts_native(
            dataset_meta.tasks, cosmos_model, device=device, dtype=torch.bfloat16,
        )
    else:
        if is_main:
            logging.info("Encoding shared empty prompt ...")
        shared_prompt_embeds = _encode_empty_prompt_native(cosmos_model, dtype=torch.bfloat16)

    # Free text encoder after all prompts are encoded.
    if cosmos_model.text_encoder is not None:
        if hasattr(cosmos_model.text_encoder, "model") and cosmos_model.text_encoder.model is not None:
            cosmos_model.text_encoder.model = cosmos_model.text_encoder.model.to("cpu")
        del cosmos_model.text_encoder
        cosmos_model.text_encoder = None
        torch.cuda.empty_cache()
        if is_main:
            logging.info("Freed text encoder; proceeding with VAE + DiT only.")

    # ── Output paths ──────────────────────────────────────────────────────────
    output_dir = str(config.assets_dirs / data_config.repo_id)
    os.makedirs(output_dir, exist_ok=True)
    frames_tag = "f" + "_".join(str(f) for f in data_config.video_delta_frames)
    keys_tag = "+".join(cam_names)
    tau_tag = f"{int(round(args.sigma * 1000)):04d}"
    tag = args.output_tag

    tmp_dirs: Dict[int, str] = {}
    for bi in block_indices:
        td = os.path.join(
            output_dir,
            f"_{tag}_tmp__{keys_tag}__{frames_tag}__blk{bi}_t{tau_tag}",
        )
        os.makedirs(td, exist_ok=True)
        tmp_dirs[bi] = td

    def _out_prefix_for_block(bi: int) -> str:
        return os.path.join(
            output_dir,
            f"{tag}_cache__{keys_tag}__{frames_tag}__blk{bi}_t{tau_tag}",
        )

    # ── Shard indices across ranks ────────────────────────────────────────────
    all_indices = list(range(len(dataset)))
    shard_indices = all_indices[rank::world_size]

    per_block_done: List[set] = []
    for bi in block_indices:
        per_block_done.append(_scan_existing_keys(tmp_dirs[bi], rank))
    done_indices = set.intersection(*per_block_done) if per_block_done else set()
    todo_indices = [i for i in shard_indices if i not in done_indices]

    if is_main:
        logging.info(f"Total {len(dataset)} samples, {world_size} GPUs, ~{len(shard_indices)} per GPU")
    if done_indices:
        logging.info(f"[rank {rank}] Resuming: {len(done_indices)} done, {len(todo_indices)} remaining")

    pool_size = tuple(args.pool_size)
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + rank)

    # ── Extract ───────────────────────────────────────────────────────────────
    chunk_id = len(glob.glob(os.path.join(tmp_dirs[block_indices[0]], f"rank{rank}_chunk*.pt")))
    pending_bufs: Dict[int, dict] = {bi: {} for bi in block_indices}
    pbar = tqdm.tqdm(todo_indices, desc=f"[rank {rank}] Extracting", disable=not is_main)
    for idx in pbar:
        sample = dataset[idx]

        if task_embs is not None:
            task_idx = int(sample["task_index"])
            sample_text_emb = task_embs[task_idx].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        elif shared_prompt_embeds is not None:
            sample_text_emb = shared_prompt_embeds.to(device=device, dtype=torch.bfloat16)
            if sample_text_emb.ndim == 2:
                sample_text_emb = sample_text_emb.unsqueeze(0)
        else:
            sample_text_emb = torch.zeros(1, 512, net.model_channels, device=device, dtype=torch.bfloat16)

        per_cam_all: Dict[str, Dict[int, torch.Tensor]] = {}
        for video_key, cam_name in zip(data_config.video_image_keys, cam_names):
            frames = np.asarray(sample[video_key])
            vid = _prepare_video(frames, target_hw=args.resolution).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clean_latent = cosmos_model.encode(vid)

                multi_feats = _extract_dit_features_multi(
                    clean_latent_5d=clean_latent,
                    cosmos_model=cosmos_model,
                    sigma=args.sigma,
                    text_embeddings=sample_text_emb,
                    block_indices=block_indices,
                    pool_size=pool_size,
                    device=device,
                    dtype=torch.bfloat16,
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

    # vid2world.cleanup() only destroys context-parallel groups (unused here).
    # We manage our own DDP lifecycle.
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
