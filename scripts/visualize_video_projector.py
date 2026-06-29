#!/usr/bin/env python3
"""Visualize a trained video_projector by decoding its predicted VAE latents.

Pipeline (per sample, per camera):

    Image (224×224)  -- SigLIP / PaliGemma --> VLA hidden @ layer N  [B, 256, 2048]
                                                        │
                                                        ▼
                                    video_projector.project_vla
                                                        │
                                                        ▼
                                  predicted latents [B, 256, T'=5, C=48]
                                                        │ reshape + spatial upsample
                                                        ▼
                                       latents [B, 48, 5, 28, 28]
                                                        │ Wan2.2 VAE decoder
                                                        ▼
                                       video [B, 3, 17, 224, 224]
                                                        │ save mp4
                                                        ▼

Usage::

    python scripts/visualize_video_projector.py \
        --config-name pi0_libero_video_layer11_5e-1_wanvae0-16_multi_frame_concat_visual \
        --ckpt-dir checkpoints/.../30000 \
        --vae-dir /path/to/Wan2.2-TI2V-5B \
        --num-tasks 8 \
        --frames-per-task 4 \
        --cameras image \
        --output-dir ./viz_outputs \
        --save-comparison

Sample selection
----------------
We pick **distinct lerobot episodes** (each libero episode corresponds to a
single task instance), then sample ``--frames-per-task`` frames uniformly
across each episode's timeline (start, mid, late, ...).

  --num-tasks N         number of episodes to visualise (default 8)
  --frames-per-task K   how many frames to sample per episode (default 4,
                        evenly spaced across the episode)
  --episode-stride S    pick episodes ``[start_episode, +S, +2S, ...]``
                        Use a large stride to cover diverse tasks.
  --start-episode E     first episode to consider (default 0)

  --cameras image       only render this set of cameras (comma-separated;
                        names follow ``video_image_keys``, typically
                        ``image`` and/or ``wrist_image``). Default is
                        ``image`` only.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import jax
import safetensors.torch
import torch
import torch.nn.functional as F
import torchvision.io as tv_io
import torchvision.utils as tv_utils

import openpi.models.model as _model
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.video_projector import create_video_align_projector


# ────────────────────────────────────────────────────────────────────────────
# WAN VAE loader
# ────────────────────────────────────────────────────────────────────────────

def _load_wan_vae_class():
    workspace_root = Path(__file__).resolve().parents[2]
    candidates = [
        workspace_root / "RoboTwin/policy/pi05/src/videomodel/Wan2.2/wan/modules/vae2_2.py",
        workspace_root / "starVLA/starVLA/videomodel/Wan2.2/wan/modules/vae2_2.py",
        Path("/mnt/data/mqs/workspace/VLA/RoboTwin/policy/pi05/src/videomodel/Wan2.2/wan/modules/vae2_2.py"),
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("vae2_2_standalone", str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            logging.info(f"Loaded Wan2_2_VAE from {path}")
            return mod.Wan2_2_VAE
    raise FileNotFoundError("Cannot locate Wan2.2 vae2_2.py.")


# ────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────

def _save_video_chw_t(video: torch.Tensor, out_path: Path, fps: int = 8) -> None:
    """Save a [3, T, H, W] tensor in [-1, 1] as an mp4."""
    v = video.clamp(-1, 1).float()
    v = ((v + 1.0) / 2.0 * 255.0).to(torch.uint8)
    v = v.permute(1, 2, 3, 0).contiguous().cpu()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tv_io.write_video(str(out_path), v, fps=fps)


def _save_grid_png(video: torch.Tensor, out_path: Path) -> None:
    """Save a horizontal strip of frames from a [3, T, H, W] tensor in [-1, 1]."""
    v = video.clamp(-1, 1).float()
    v = ((v + 1.0) / 2.0)
    v = v.permute(1, 0, 2, 3).contiguous().cpu()
    grid = tv_utils.make_grid(v, nrow=v.shape[0], padding=2, pad_value=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tv_utils.save_image(grid, str(out_path))


def _save_comparison_video(
    pred_video: torch.Tensor,
    gt_video: torch.Tensor | None,
    input_image: torch.Tensor | None,
    out_path: Path,
    fps: int = 8,
    layout: str = "gt_pred",
) -> None:
    """Save a side-by-side mp4. Layouts:

      "gt_pred"        : [GT | pred]                  (default)
      "input_gt_pred"  : [input | GT | pred]
      "pred_only"      : [pred]

    pred_video : [3, T, H, W] in [-1, 1]   (required)
    gt_video   : [3, T, H, W] in [-1, 1]   (optional)
    input_image: [3, H, W]    in [-1, 1]   (optional, broadcast over T)
    """
    T = pred_video.shape[1]
    H, W = pred_video.shape[2], pred_video.shape[3]

    panels = []
    panel_labels = []

    if layout == "input_gt_pred" and input_image is not None:
        inp = input_image.clamp(-1, 1).float()
        if inp.shape[-2:] != (H, W):
            inp = F.interpolate(inp.unsqueeze(0), size=(H, W), mode="bilinear",
                                align_corners=False).squeeze(0)
        panels.append(inp.unsqueeze(1).expand(3, T, H, W))
        panel_labels.append("input")

    if layout in ("gt_pred", "input_gt_pred") and gt_video is not None:
        g = gt_video.clamp(-1, 1).float()
        if g.shape[1] != T or g.shape[-2:] != (H, W):
            g = F.interpolate(g.unsqueeze(0), size=(T, H, W), mode="trilinear",
                              align_corners=False).squeeze(0)
        panels.append(g)
        panel_labels.append("GT")

    panels.append(pred_video.clamp(-1, 1).float())
    panel_labels.append("pred")

    label_h = 8
    label_colours = {
        "input": (0.10, 0.55, 0.95),
        "GT":    (0.20, 0.80, 0.30),
        "pred":  (0.95, 0.50, 0.20),
    }
    decorated = []
    for vid, label in zip(panels, panel_labels, strict=True):
        bar = torch.zeros(3, T, label_h, vid.shape[-1], dtype=vid.dtype, device=vid.device)
        c = label_colours.get(label, (0.5, 0.5, 0.5))
        for ch, val in enumerate(c):
            bar[ch] = val * 2.0 - 1.0
        decorated.append(torch.cat([bar, vid], dim=2))

    combined = torch.cat(decorated, dim=3)
    _save_video_chw_t(combined, out_path, fps=fps)


# ────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _prefix_hidden_at_layer(
    model: pi0_pytorch.PI0Pytorch,
    observation,
    align_layer: int,
    device: torch.device,
):
    images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(  # noqa: SLF001
        observation, train=False
    )

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )

    if (
        model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        == torch.bfloat16
    ):
        prefix_embs = prefix_embs.to(torch.bfloat16)

    att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    att_2d_4d = model._prepare_attention_masks_4d(att_2d)  # noqa: SLF001
    target_dtype = prefix_embs.dtype
    if att_2d_4d.dtype != target_dtype:
        att_2d_4d = torch.where(
            att_2d_4d == 0,
            torch.zeros((), dtype=target_dtype, device=att_2d_4d.device),
            torch.full((), torch.finfo(target_dtype).min, dtype=target_dtype, device=att_2d_4d.device),
        )
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    out = model.paligemma_with_expert.paligemma.language_model(
        inputs_embeds=prefix_embs,
        attention_mask=att_2d_4d,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        adarms_cond=None,
        output_hidden_states=True,
    )
    align_hidden = out.hidden_states[align_layer + 1]

    num_cam_slots = len(images)
    img_len = prefix_embs.shape[1] - lang_masks.shape[1]
    tokens_per_cam = img_len // num_cam_slots
    return align_hidden, images, img_masks, tokens_per_cam, num_cam_slots


@torch.no_grad()
def _project_to_latents(
    projector,
    cam_vla_tokens: torch.Tensor,
    *,
    num_frames: int,
    vae_dim: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Run ``project_vla`` and reshape to [B, C, T', H, W] for VAE decode."""
    B, N, _ = cam_vla_tokens.shape
    side = int(N ** 0.5)
    if side * side != N:
        raise ValueError(f"tokens_per_cam={N} is not a perfect square.")

    pred = projector.project_vla(cam_vla_tokens.float())
    pred = pred.reshape(B, N, num_frames, vae_dim)
    pred = pred.permute(0, 3, 2, 1).contiguous()
    pred = pred.reshape(B, vae_dim, num_frames, side, side)

    if (target_h, target_w) != (side, side):
        pred = pred.permute(0, 2, 1, 3, 4).reshape(B * num_frames, vae_dim, side, side)
        pred = F.interpolate(pred, size=(target_h, target_w), mode="bilinear", align_corners=False)
        pred = pred.reshape(B, num_frames, vae_dim, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()

    return pred


def _match_magnitude(
    pred: torch.Tensor,
    gt: torch.Tensor | None,
    mode: str,
    constant: float,
) -> torch.Tensor:
    if mode == "none":
        return pred
    if mode == "constant":
        norm = pred.float().norm(dim=1, keepdim=True).clamp(min=1e-6)
        return pred / norm * constant
    if mode == "gt":
        if gt is None:
            logging.warning("--match-magnitude=gt requested but no GT cache available; falling back to 'none'.")
            return pred
        gt_f = gt.to(device=pred.device, dtype=torch.float32)
        if gt_f.shape != pred.shape:
            B, C, T_g, H_g, W_g = gt_f.shape
            gt_flat = gt_f.permute(0, 2, 1, 3, 4).reshape(B * T_g, C, H_g, W_g)
            gt_flat = F.interpolate(gt_flat, size=(pred.shape[3], pred.shape[4]),
                                    mode="bilinear", align_corners=False)
            gt_f = gt_flat.reshape(B, T_g, C, pred.shape[3], pred.shape[4]).permute(0, 2, 1, 3, 4)
        gt_norm = gt_f.norm(dim=1, keepdim=True).clamp(min=1e-6)
        pred_norm = pred.float().norm(dim=1, keepdim=True).clamp(min=1e-6)
        return pred.float() / pred_norm * gt_norm
    raise ValueError(f"Unknown match-magnitude mode: {mode!r}")


@torch.no_grad()
def _decode_with_wan(vae, latent: torch.Tensor) -> torch.Tensor:
    z = latent.to(device=vae.device, dtype=vae.dtype)
    with torch.amp.autocast("cuda", dtype=vae.dtype):
        vae.model.clear_cache()
        video = vae.model.decode(z.unsqueeze(0), vae.scale)
    return video[0].float().clamp(-1, 1)


def _diagnose_pred_vs_gt(pred_raw: torch.Tensor, gt: torch.Tensor) -> dict:
    p = pred_raw.float()
    g = gt.float().to(p.device)
    cos = (F.normalize(p, dim=1) * F.normalize(g, dim=1)).sum(dim=1)
    p_norm = p.norm(dim=1).clamp(min=1e-6)
    g_norm = g.norm(dim=1).clamp(min=1e-6)
    ratio = p_norm / g_norm
    log_ratio = ratio.log()
    return {
        "cos_sim": float(cos.mean().item()),
        "norm_ratio": float(ratio.mean().item()),
        "norm_log_std": float(log_ratio.std().item()),
    }


def _decode_prompt(tokenizer, token_ids: torch.Tensor) -> str:
    try:
        ids = [int(x) for x in token_ids.tolist() if int(x) != 0]
        sp = getattr(tokenizer, "_tokenizer", tokenizer)
        return sp.decode(ids).strip()
    except Exception:
        return "<decode failed>"


def _safe_filename(s: str, maxlen: int = 60) -> str:
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in s.strip())
    return s[:maxlen]


# ────────────────────────────────────────────────────────────────────────────
# Sample selection — pick (episode, frame_within_episode) tuples
# ────────────────────────────────────────────────────────────────────────────

def _select_frame_indices(
    repo_id: str,
    *,
    num_tasks: int,
    frames_per_task: int,
    start_episode: int,
    episode_stride: int,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Pick a list of global frame indices spanning multiple episodes.

    Returns
    -------
    indices  : list[int]
        Global frame indices into the underlying LeRobotDataset.
    metadata : list[tuple[int, int, int]]
        Aligned with ``indices``; each entry is ``(episode_idx, frame_in_episode, episode_length)``.
    """
    import lerobot.common.datasets.lerobot_dataset as _ld

    meta = _ld.LeRobotDatasetMetadata(repo_id)
    total_eps = int(meta.total_episodes)
    if total_eps == 0:
        raise RuntimeError(f"Dataset {repo_id} has no episodes")

    # Pre-compute cumulative offsets so we can map (episode, offset) → global frame index.
    ep_lengths = [int(meta.episodes[i]["length"]) for i in range(total_eps)]
    cum_offsets = [0]
    for L in ep_lengths:
        cum_offsets.append(cum_offsets[-1] + L)

    # Pick episodes
    picked = list(range(start_episode, total_eps, max(1, episode_stride)))[:num_tasks]
    if not picked:
        raise RuntimeError(
            f"No episodes selected. Check --start-episode={start_episode}, "
            f"--episode-stride={episode_stride}, total={total_eps}"
        )

    indices = []
    metadata = []
    for ep in picked:
        L = ep_lengths[ep]
        if L < 1:
            continue
        for k in range(frames_per_task):
            # Evenly sample positions inside the episode (e.g. 4 → at 0.125, 0.375, 0.625, 0.875 of the way)
            # but always include the very last frame too if frames_per_task > 1.
            if frames_per_task == 1:
                offset = 0
            else:
                offset = int((k + 0.5) * (L - 1) / frames_per_task)
            offset = min(offset, L - 1)
            indices.append(cum_offsets[ep] + offset)
            metadata.append((ep, offset, L))

    return indices, metadata


def _build_dataset_for_inference(config):
    """Recreate the same transformed dataset that ``create_data_loader`` would,
    but expose it as an indexable object so we can pick samples ourselves.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    vae_cache = None
    if config.video_cache_path is not None and not getattr(config, "mmap_video_cache", False):
        logging.info(f"Loading VAE cache from {config.video_cache_path} ...")
        vae_cache = torch.load(config.video_cache_path, map_location="cpu", weights_only=False)
        logging.info(f"Loaded {len(vae_cache)} cached VAE features")

    base_dataset = _data.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    transformed = _data.transform_dataset(
        base_dataset, data_config, skip_norm_stats=False, vae_cache=vae_cache,
    )
    return transformed, data_config


def _collate_indices(dataset, indices: list[int]):
    """Collate a list of dataset[idx] dicts into a single batched dict (numpy)."""
    items = [dataset[i] for i in indices]
    return _data._collate_fn(items)  # noqa: SLF001


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", required=True, help="TrainConfig name")
    parser.add_argument("--ckpt-dir", required=True,
                        help="Path to checkpoint step dir (model.safetensors + video_projector.safetensors)")
    parser.add_argument("--vae-dir", required=True, help="Directory containing Wan2.2_VAE.pth")

    # Sample selection
    parser.add_argument("--num-tasks", type=int, default=8,
                        help="Number of distinct episodes (≈ tasks for libero) to visualise.")
    parser.add_argument("--frames-per-task", type=int, default=4,
                        help="How many frames to sample uniformly within each picked episode.")
    parser.add_argument("--start-episode", type=int, default=0,
                        help="First episode index to consider.")
    parser.add_argument("--episode-stride", type=int, default=1,
                        help="Pick episodes [start, +S, +2S, ...]. Use a large S to scatter "
                             "across all tasks (libero has hundreds of episodes total).")

    # Camera filtering
    parser.add_argument("--cameras", type=str, default="image",
                        help="Comma-separated camera keys to render. Names follow "
                             "video_image_keys (typically 'image' and 'wrist_image'). "
                             "Default: 'image' only (skips wrist).")

    # Output
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--match-magnitude", choices=("none", "gt", "constant"), default="gt")
    parser.add_argument("--mag-constant", type=float, default=4.0)
    parser.add_argument("--save-input-image", action="store_true",
                        help="Also save the input image used to drive the prediction")
    parser.add_argument("--save-comparison", action="store_true",
                        help="Save a side-by-side mp4 combining GT and prediction.")
    parser.add_argument("--comparison-layout", choices=("gt_pred", "input_gt_pred"), default="gt_pred")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="How many frames to batch together when running the model.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    logging.getLogger().setLevel(logging.INFO)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_cameras = {c.strip() for c in args.cameras.split(",") if c.strip()}
    logging.info(f"Camera filter: {sorted(allowed_cameras)}")

    # ── 1. Load training config ─────────────────────────────────────────────
    config = _config.get_config(args.config_name)
    logging.info(f"Loaded TrainConfig: {config.name}")

    ckpt_dir = Path(args.ckpt_dir)
    if not (ckpt_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"{ckpt_dir / 'model.safetensors'} not found")
    if not (ckpt_dir / "video_projector.safetensors").is_file():
        raise FileNotFoundError(
            f"{ckpt_dir / 'video_projector.safetensors'} not found "
            "(this checkpoint was saved before video_projector saving was enabled)"
        )

    # ── 2. Build PI0Pytorch and load weights ────────────────────────────────
    logging.info("Building PI0Pytorch model ...")
    object.__setattr__(config.model, "dtype", "bfloat16")
    model = pi0_pytorch.PI0Pytorch(config.model).to(device)
    safetensors.torch.load_model(model, str(ckpt_dir / "model.safetensors"))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()

    # ── 3. Build & load video_projector ─────────────────────────────────────
    bc = getattr(config.data, "base_config", None)
    delta_frames = getattr(bc, "video_delta_frames", None)
    delta_frames_aux = getattr(bc, "video_delta_frames_aux", None)
    video_image_keys = list(getattr(bc, "video_image_keys", None) or ["image"])
    num_cameras = len(video_image_keys)

    vla_dim = model.paligemma_with_expert.paligemma.config.text_config.hidden_size
    projector = create_video_align_projector(
        config.video_align_mode, vla_dim, config.video_feat_dim,
        video_delta_frames=delta_frames,
        video_delta_frames_aux=delta_frames_aux,
        use_vla_norm=config.use_vla_norm,
        num_cameras=num_cameras,
        loss_weight_primary=getattr(config, "video_loss_weight_primary", 1.0),
        loss_weight_aux=getattr(config, "video_loss_weight_aux", 1.0),
        contrast_weight=getattr(config, "video_contrast_weight", 0.1),
        temperature=getattr(config, "video_contrast_temperature", 0.07),
    ).to(device)
    safetensors.torch.load_model(projector, str(ckpt_dir / "video_projector.safetensors"))
    projector.eval()

    if not hasattr(projector, "project_vla"):
        raise NotImplementedError(
            f"Projector {type(projector).__name__} does not expose .project_vla(); "
            "extend this script to support it."
        )
    num_frames = projector.num_frames
    vae_dim = projector.vae_dim
    logging.info(
        f"Projector {type(projector).__name__}  |  out: T'={num_frames}, C={vae_dim}, "
        f"vla_dim={vla_dim}, align_layer={config.vla_align_layer}"
    )

    # ── 4. Load WAN VAE ─────────────────────────────────────────────────────
    Wan2_2_VAE = _load_wan_vae_class()
    vae_pth = os.path.join(args.vae_dir, "Wan2.2_VAE.pth")
    if not os.path.isfile(vae_pth):
        raise FileNotFoundError(f"{vae_pth} not found")
    vae = Wan2_2_VAE(vae_pth=vae_pth, device=device, dtype=dtype)
    vae.model.eval()

    prompt_tokenizer = None
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        prompt_tokenizer = PaligemmaTokenizer(max_len=config.model.max_token_len)
    except Exception as e:
        logging.info(f"(prompt decoding disabled: {e})")

    # ── 5. Build (transformed) dataset and select frame indices ─────────────
    logging.info("Building dataset for inference ...")
    dataset, data_config = _build_dataset_for_inference(config)

    indices, idx_metadata = _select_frame_indices(
        data_config.repo_id,
        num_tasks=args.num_tasks,
        frames_per_task=args.frames_per_task,
        start_episode=args.start_episode,
        episode_stride=args.episode_stride,
    )
    logging.info(
        f"Selected {len(indices)} frames over "
        f"{len({m[0] for m in idx_metadata})} episodes "
        f"(num_tasks={args.num_tasks} × frames_per_task={args.frames_per_task})"
    )

    align_layer = config.vla_align_layer
    if align_layer is None or align_layer < 0:
        raise ValueError(f"vla_align_layer must be >= 0, got {align_layer}")

    # ── 6. Iterate manually-batched samples ─────────────────────────────────
    saved_frames = 0
    for batch_start in range(0, len(indices), args.batch_size):
        batch_indices = indices[batch_start:batch_start + args.batch_size]
        batch_metadata = idx_metadata[batch_start:batch_start + args.batch_size]

        # Collate into batched numpy dict and convert to torch tensors.
        try:
            batch = _collate_indices(dataset, batch_indices)
        except Exception as e:
            logging.warning(f"Failed to load batch starting at idx {batch_start} ({batch_indices}): {e}")
            continue
        batch = jax.tree.map(torch.as_tensor, batch)
        observation = _model.Observation.from_dict(batch)

        observation = jax.tree.map(lambda x: x.to(device), observation)
        vae_cache = getattr(observation, "vae_cache", None)

        # Decode prompts for filenames + logging.
        prompts_per_item = []
        if prompt_tokenizer is not None and observation.tokenized_prompt is not None:
            for i in range(observation.tokenized_prompt.shape[0]):
                p = _decode_prompt(prompt_tokenizer, observation.tokenized_prompt[i].cpu())
                prompts_per_item.append(p)

        align_hidden, images, img_masks, tokens_per_cam, num_cam_slots = _prefix_hidden_at_layer(
            model, observation, align_layer, device
        )

        if vae_cache is not None:
            if config.video_cache_layout == "TCHW":
                vae_cache = vae_cache.permute(0, 1, 3, 2, 4, 5).contiguous()
            target_h, target_w = vae_cache.shape[-2], vae_cache.shape[-1]
            n_vae_cams = vae_cache.shape[1]
        else:
            target_h = target_w = int(tokens_per_cam ** 0.5)
            n_vae_cams = 0

        valid_cam_indices = [i for i, m in enumerate(img_masks) if bool(m.any())]

        # Project + rescale once per (kept) cam for the whole batch.
        cam_pred: dict[int, torch.Tensor] = {}
        cam_gt: dict[int, torch.Tensor | None] = {}
        cam_names: dict[int, str] = {}
        for vae_idx in range(max(n_vae_cams, len(valid_cam_indices))):
            if vae_idx >= len(valid_cam_indices):
                break
            cam_name = video_image_keys[vae_idx] if vae_idx < len(video_image_keys) else f"cam{vae_idx}"
            if cam_name not in allowed_cameras:
                continue

            vla_cam_idx = valid_cam_indices[vae_idx]
            cam_start = vla_cam_idx * tokens_per_cam
            cam_end = cam_start + tokens_per_cam
            cam_vla = align_hidden[:, cam_start:cam_end, :]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_latent_raw = _project_to_latents(
                    projector, cam_vla,
                    num_frames=num_frames, vae_dim=vae_dim,
                    target_h=target_h, target_w=target_w,
                )

            gt_latent = None
            if vae_cache is not None and vae_idx < vae_cache.shape[1]:
                gt_latent = vae_cache[:, vae_idx]

            if gt_latent is not None:
                diag = _diagnose_pred_vs_gt(pred_latent_raw, gt_latent)
                logging.info(
                    f"  [{cam_name}] cos_sim={diag['cos_sim']:.3f}  "
                    f"norm_ratio={diag['norm_ratio']:.3f}  "
                    f"norm_log_std={diag['norm_log_std']:.3f}"
                )

            pred_latent = _match_magnitude(
                pred_latent_raw, gt_latent,
                mode=args.match_magnitude, constant=args.mag_constant,
            )
            cam_pred[vae_idx] = pred_latent
            cam_gt[vae_idx] = gt_latent
            cam_names[vae_idx] = cam_name

        if not cam_pred:
            logging.warning(f"No cameras matched --cameras={args.cameras!r}; check the spelling.")
            return

        # Save per-frame outputs for every cam we kept.
        B = align_hidden.shape[0]
        for b in range(B):
            ep_idx, frame_in_ep, ep_len = batch_metadata[b]
            prompt_short = prompts_per_item[b] if b < len(prompts_per_item) else ""
            prompt_tag = "__" + _safe_filename(prompt_short, maxlen=40) if prompt_short else ""
            frame_tag = f"ep{ep_idx:04d}_f{frame_in_ep:04d}of{ep_len:04d}{prompt_tag}"

            for vae_idx, pred_latent in cam_pred.items():
                cam_name = cam_names[vae_idx]
                tag = f"{frame_tag}__{cam_name}"

                pred_video = _decode_with_wan(vae, pred_latent[b])
                _save_video_chw_t(pred_video, out_dir / "predicted" / f"{tag}.mp4", fps=args.fps)
                _save_grid_png(pred_video, out_dir / "predicted_grid" / f"{tag}.png")

                gt_video = None
                gt_latent = cam_gt.get(vae_idx)
                if gt_latent is not None:
                    gt_video = _decode_with_wan(vae, gt_latent[b])
                    _save_video_chw_t(gt_video, out_dir / "gt" / f"{tag}.mp4", fps=args.fps)
                    _save_grid_png(gt_video, out_dir / "gt_grid" / f"{tag}.png")

                input_img = None
                vla_cam_idx = valid_cam_indices[vae_idx]
                if args.save_input_image or (
                    args.save_comparison and args.comparison_layout == "input_gt_pred"
                ):
                    input_img = images[vla_cam_idx][b].clamp(-1, 1).float()
                if args.save_input_image:
                    out_dir.joinpath("input_images").mkdir(parents=True, exist_ok=True)
                    tv_utils.save_image(((input_img + 1.0) / 2.0).cpu(),
                                        str(out_dir / "input_images" / f"{tag}.png"))

                if args.save_comparison:
                    _save_comparison_video(
                        pred_video=pred_video,
                        gt_video=gt_video,
                        input_image=input_img,
                        out_path=out_dir / "comparison" / f"{tag}.mp4",
                        fps=args.fps,
                        layout=args.comparison_layout,
                    )

            saved_frames += 1
            logging.info(
                f"[{saved_frames}/{len(indices)}] saved ep={ep_idx} frame={frame_in_ep}/{ep_len}"
                + (f" prompt={prompt_short!r}" if prompt_short else "")
            )

    logging.info(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
