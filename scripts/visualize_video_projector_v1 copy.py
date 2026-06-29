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
        --ckpt-dir checkpoints/pi0_libero_video_layer11_5e-1_wanvae0-16_multi_frame_concat_visual/pi0_libero_video_layer11_5e-1_wanvae0-16_multi_frame_concat_visual_4gpu/30000 \
        --vae-dir /path/to/Wan2.2-TI2V-5B \
        --num-samples 4 \
        --output-dir ./viz_outputs

Notes
-----
* The projector is trained with **cosine** loss, which constrains direction but
  not magnitude. We optionally rescale predicted latents to match the per-token
  L2 norm of GT cached latents (``--match-magnitude=gt``) so that the WAN VAE
  decoder produces a recognisable image. ``--match-magnitude=none`` shows the
  raw output.
* When the dataset sample has ``vae_cache``, we also decode the GT latents and
  save them side-by-side for comparison.
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

import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.video_projector import create_video_align_projector


# ────────────────────────────────────────────────────────────────────────────
# WAN VAE loader — search known repo locations for the standalone vae2_2.py
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
    raise FileNotFoundError(
        "Cannot locate Wan2.2 vae2_2.py. Pass a custom path via PYTHONPATH or "
        "edit `_load_wan_vae_class` candidates."
    )


# ────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────

def _save_video_chw_t(video: torch.Tensor, out_path: Path, fps: int = 8) -> None:
    """Save a [3, T, H, W] tensor in [-1, 1] as an mp4."""
    v = video.clamp(-1, 1).float()
    v = ((v + 1.0) / 2.0 * 255.0).to(torch.uint8)
    v = v.permute(1, 2, 3, 0).contiguous().cpu()  # [T, H, W, 3]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tv_io.write_video(str(out_path), v, fps=fps)


def _save_grid_png(video: torch.Tensor, out_path: Path) -> None:
    """Save a horizontal strip of frames from a [3, T, H, W] tensor in [-1, 1]."""
    v = video.clamp(-1, 1).float()
    v = ((v + 1.0) / 2.0)  # [0, 1]
    v = v.permute(1, 0, 2, 3).contiguous().cpu()  # [T, 3, H, W]
    grid = tv_utils.make_grid(v, nrow=v.shape[0], padding=2, pad_value=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tv_utils.save_image(grid, str(out_path))


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
    """Run prefix-only (image+language) forward and return hidden states at
    ``align_layer``.

    Returns
    -------
    align_hidden     : Tensor [B, L_prefix, D]
        Hidden state output of layer ``align_layer``.
    images           : list[Tensor]
        Pre-processed images per camera slot (post-resize, in [-1, 1]).
    img_masks        : list[Tensor]
        Per-camera image validity mask (some slots may be padded zeros).
    tokens_per_cam   : int
        Number of image tokens per camera slot.
    num_cam_slots    : int
        Total camera slots in the model (e.g. 3 for libero — incl. padding).
    """
    images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(  # noqa: SLF001
        observation, train=False
    )

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )

    # Some weights live in bfloat16; align dtype.
    if (
        model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        == torch.bfloat16
    ):
        prefix_embs = prefix_embs.to(torch.bfloat16)

    att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    att_2d_4d = model._prepare_attention_masks_4d(att_2d)  # noqa: SLF001
    # SDPA in PyTorch >= 2.3 requires the attention bias dtype to match query's dtype.
    # Replace the (very negative) sentinel with finfo.min for the target dtype to
    # avoid losing range when casting from fp32 to bf16/fp16.
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
    # hidden_states[0] is the embedding output, hidden_states[k+1] is the output of layer k.
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
    """Run ``project_vla`` and reshape to [B, C, T', H, W] for VAE decode.

    cam_vla_tokens : [B, N=tokens_per_cam, D]
    returns        : [B, C=vae_dim, T'=num_frames, H=target_h, W=target_w]
    """
    B, N, _ = cam_vla_tokens.shape
    side = int(N ** 0.5)
    if side * side != N:
        raise ValueError(f"tokens_per_cam={N} is not a perfect square; cannot reshape to a 2D grid.")

    pred = projector.project_vla(cam_vla_tokens.float())            # [B, N, T*C]
    pred = pred.reshape(B, N, num_frames, vae_dim)                   # [B, N, T', C]
    pred = pred.permute(0, 3, 2, 1).contiguous()                     # [B, C, T', N]
    pred = pred.reshape(B, vae_dim, num_frames, side, side)          # [B, C, T', side, side]

    if (target_h, target_w) != (side, side):
        # Spatially upscale to the original VAE latent resolution (e.g. 16 → 28)
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
    """Optionally rescale ``pred`` so its L2 magnitude matches ``gt`` per token.

    pred / gt: [B, C, T', H, W]
    """
    if mode == "none":
        return pred

    if mode == "constant":
        # Scale predicted so that its mean per-token (over C) norm equals ``constant``
        norm = pred.float().norm(dim=1, keepdim=True).clamp(min=1e-6)
        return pred / norm * constant

    if mode == "gt":
        if gt is None:
            logging.warning("--match-magnitude=gt requested but no GT cache available; falling back to 'none'.")
            return pred
        gt_f = gt.to(device=pred.device, dtype=torch.float32)
        if gt_f.shape != pred.shape:
            # _project_to_latents should already have matched spatial size, but
            # guard against accidental mismatches by interpolating GT here.
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
    """Decode a single batch element [C, T', H, W] → video [3, T_video, H_img, W_img]."""
    z = latent.to(device=vae.device, dtype=vae.dtype)
    with torch.amp.autocast("cuda", dtype=vae.dtype):
        vae.model.clear_cache()
        video = vae.model.decode(z.unsqueeze(0), vae.scale)  # [1, 3, T, H, W]
    return video[0].float().clamp(-1, 1)


def _diagnose_pred_vs_gt(pred_raw: torch.Tensor, gt: torch.Tensor) -> dict:
    """Compare predicted (pre-rescale) latents to GT.

    pred_raw, gt : [B, C, T', H, W] (assumed same shape)

    Returns per-batch-mean diagnostics:
      cos_sim       : channel-direction agreement (training objective)
      norm_ratio    : ||pred|| / ||gt||  (1.0 = perfect magnitude)
      norm_log_std  : std of log(norm_ratio) — how spread-out the magnitudes are
    """
    p = pred_raw.float()
    g = gt.float().to(p.device)
    cos = (F.normalize(p, dim=1) * F.normalize(g, dim=1)).sum(dim=1)            # [B, T', H, W]
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
    """Best-effort prompt decoding for logging.

    ``tokenizer`` is expected to expose a sentencepiece processor at
    ``tokenizer._tokenizer`` (matches openpi's PaligemmaTokenizer wrapper).
    """
    try:
        ids = [int(x) for x in token_ids.tolist() if int(x) != 0]
        sp = getattr(tokenizer, "_tokenizer", tokenizer)
        return sp.decode(ids).strip()
    except Exception:
        return "<decode failed>"


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", required=True, help="TrainConfig name registered in openpi.training.config")
    parser.add_argument("--ckpt-dir", required=True,
                        help="Path to checkpoint step directory (containing model.safetensors and video_projector.safetensors)")
    parser.add_argument("--vae-dir", required=True, help="Directory containing Wan2.2_VAE.pth")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--start-batch", type=int, default=0,
                        help="Skip this many batches before visualising")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--match-magnitude", choices=("none", "gt", "constant"), default="gt")
    parser.add_argument("--mag-constant", type=float, default=4.0)
    parser.add_argument("--save-input-image", action="store_true",
                        help="Also save the input image used to drive the prediction")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override config.batch_size (use a small value to save GPU memory)")
    args = parser.parse_args()

    # NOTE: lerobot calls ``logging.basicConfig`` at import time which locks the root
    # logger to WARNING; ``force=True`` clears its handlers so our INFO logs show up.
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

    # ── 3. Build & load video_projector ────────────────────────────────────
    bc = getattr(config.data, "base_config", None)
    delta_frames = getattr(bc, "video_delta_frames", None)
    delta_frames_aux = getattr(bc, "video_delta_frames_aux", None)
    video_image_keys = getattr(bc, "video_image_keys", None)
    num_cameras = len(video_image_keys) if video_image_keys else 1

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
            "add an analogous inference helper or extend this script."
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

    # Try to grab a tokenizer for prompt decoding (best-effort).
    prompt_tokenizer = None
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        prompt_tokenizer = PaligemmaTokenizer(max_len=config.model.max_token_len)
    except Exception as e:
        logging.info(f"(prompt decoding disabled: {e})")

    # ── 5. Build data loader ────────────────────────────────────────────────
    logging.info("Building data loader ...")
    # Override num_workers to 0 for simpler debugging in this single-process viz script.
    object.__setattr__(config, "num_workers", 0)
    if args.batch_size is not None:
        object.__setattr__(config, "batch_size", args.batch_size)
        logging.info(f"Overriding batch_size to {args.batch_size}")
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)

    align_layer = config.vla_align_layer
    if align_layer is None or align_layer < 0:
        raise ValueError(f"vla_align_layer must be >= 0, got {align_layer}")

    # Iterate
    saved = 0
    for batch_idx, (observation, _actions) in enumerate(loader):
        if batch_idx < args.start_batch:
            continue
        if saved >= args.num_samples:
            break

        observation = jax.tree.map(lambda x: x.to(device), observation)
        # Move VAE cache (if present) along too — preprocess_observation does not touch it.
        vae_cache = getattr(observation, "vae_cache", None)

        # Log prompt for the first item of this batch so user can correlate viz with task
        if prompt_tokenizer is not None and observation.tokenized_prompt is not None:
            first_prompt = _decode_prompt(prompt_tokenizer, observation.tokenized_prompt[0].cpu())
            logging.info(f"=== batch {batch_idx} | prompt[0]: {first_prompt!r}")

        align_hidden, images, img_masks, tokens_per_cam, num_cam_slots = _prefix_hidden_at_layer(
            model, observation, align_layer, device
        )

        # Determine GT spatial latent resolution. vae_cache shape: [B, N_cams, C, T', H', W'] (CTHW).
        if vae_cache is not None:
            if config.video_cache_layout == "TCHW":
                vae_cache = vae_cache.permute(0, 1, 3, 2, 4, 5).contiguous()
            target_h, target_w = vae_cache.shape[-2], vae_cache.shape[-1]
            n_vae_cams = vae_cache.shape[1]
        else:
            target_h = target_w = int(tokens_per_cam ** 0.5)
            n_vae_cams = 0

        # Mirror training's camera mapping: valid model cam slots → vae cache slots
        # (in the same order as ``video_image_keys``).
        valid_cam_indices = [i for i, m in enumerate(img_masks) if bool(m.any())]
        cache_cam_names = list(video_image_keys) if video_image_keys else []

        # When no vae_cache (e.g. cache disabled), still visualize all valid cams.
        n_cams_to_viz = max(n_vae_cams, len(valid_cam_indices) if vae_cache is None else n_vae_cams)

        for vae_idx in range(n_cams_to_viz):
            if vae_idx >= len(valid_cam_indices):
                break
            vla_cam_idx = valid_cam_indices[vae_idx]
            cam_name = cache_cam_names[vae_idx] if vae_idx < len(cache_cam_names) else f"cam{vae_idx}"

            cam_start = vla_cam_idx * tokens_per_cam
            cam_end = cam_start + tokens_per_cam
            cam_vla = align_hidden[:, cam_start:cam_end, :]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_latent_raw = _project_to_latents(
                    projector, cam_vla,
                    num_frames=num_frames, vae_dim=vae_dim,
                    target_h=target_h, target_w=target_w,
                )

            # GT (pre-rescale) for comparison + optional magnitude rescaling
            gt_latent_for_cam = None
            if vae_cache is not None and vae_idx < vae_cache.shape[1]:
                gt_latent_for_cam = vae_cache[:, vae_idx]  # [B, C, T', H', W']

            # Diagnostics: cos sim + norm ratio (only meaningful when GT is available)
            if gt_latent_for_cam is not None:
                diag = _diagnose_pred_vs_gt(pred_latent_raw, gt_latent_for_cam)
                logging.info(
                    f"  [{cam_name}] cos_sim={diag['cos_sim']:.3f}  "
                    f"norm_ratio={diag['norm_ratio']:.3f}  "
                    f"norm_log_std={diag['norm_log_std']:.3f}"
                )

            pred_latent = _match_magnitude(
                pred_latent_raw, gt_latent_for_cam,
                mode=args.match_magnitude, constant=args.mag_constant,
            )

            # Decode each batch element separately (VAE expects 1 sample at a time)
            B = pred_latent.shape[0]
            for b in range(B):
                if saved >= args.num_samples:
                    break
                tag = f"{batch_idx:04d}_b{b:02d}__{cam_name}"

                pred_video = _decode_with_wan(vae, pred_latent[b])
                _save_video_chw_t(pred_video, out_dir / "predicted" / f"{tag}.mp4", fps=args.fps)
                _save_grid_png(pred_video, out_dir / "predicted_grid" / f"{tag}.png")

                if gt_latent_for_cam is not None:
                    gt_video = _decode_with_wan(vae, gt_latent_for_cam[b])
                    _save_video_chw_t(gt_video, out_dir / "gt" / f"{tag}.mp4", fps=args.fps)
                    _save_grid_png(gt_video, out_dir / "gt_grid" / f"{tag}.png")

                if args.save_input_image:
                    img = images[vla_cam_idx][b].clamp(-1, 1).float()
                    img = ((img + 1.0) / 2.0)
                    out_dir.joinpath("input_images").mkdir(parents=True, exist_ok=True)
                    tv_utils.save_image(img.cpu(), str(out_dir / "input_images" / f"{tag}.png"))

                saved += 1
                logging.info(f"[{saved}/{args.num_samples}] saved {tag}")

    logging.info(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
