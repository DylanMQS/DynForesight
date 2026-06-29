"""Minimal utilities for visualizing transformer attention as per-camera heatmaps.

Pipeline (intentionally as direct as possible):
    1. take attention ``[B, H, Q, K]``, mean-reduce heads + queries, slice per
       camera and reshape to a 2-D patch grid.
    2. bilinearly resize the patch grid to the source image resolution.
    3. min-max normalize to ``[0, 1]``.
    4. apply a jet colour map.
    5. flat alpha-blend with the original image.

No border processing, gamma sharpening, percentile clipping or per-pixel alpha
modulation — just the raw heatmap rendered honestly.
"""

from __future__ import annotations

import math
import pathlib
from typing import Iterable, Mapping

import numpy as np
import torch


# ── Image / colour helpers ──────────────────────────────────────────────────


def _to_uint8_image(img: torch.Tensor) -> np.ndarray:
    """Convert a (preprocessed) image tensor to ``uint8`` ``HWC`` numpy.

    Accepts ``CHW`` or ``HWC`` layouts and assumes pixel values are roughly in
    ``[-1, 1]`` (the openpi pre-processing range).
    """
    img = img.detach().float().cpu()
    if img.ndim != 3:
        raise ValueError(f"expected a 3-D image tensor, got shape {tuple(img.shape)}")
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = img.permute(1, 2, 0)
    img = (img + 1.0) * 127.5
    return img.clamp(0, 255).numpy().astype(np.uint8)


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Map a ``[H, W]`` array in ``[0, 1]`` to a jet-like ``uint8`` ``[H, W, 3]``."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _resize_heatmap(heatmap: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    t = torch.from_numpy(heatmap.astype(np.float32))[None, None, ...]
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def overlay_heatmap(
    image_uint8: np.ndarray,
    heatmap_2d: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Resize ``heatmap_2d`` to ``image_uint8`` size, jet-colour-map and alpha-blend.

    ``overlay = (1 - alpha) * image + alpha * jet(min_max_norm(resize(heatmap)))``
    """
    h, w = image_uint8.shape[:2]
    hm = _resize_heatmap(heatmap_2d, (h, w))
    lo, hi = float(hm.min()), float(hm.max())
    hm = (hm - lo) / (hi - lo) if hi > lo else np.zeros_like(hm)
    color = _jet_colormap(hm).astype(np.float32)
    img = image_uint8.astype(np.float32)
    blended = img * (1.0 - alpha) + color * alpha
    return blended.clip(0, 255).astype(np.uint8)


def _save_png(arr: np.ndarray, path: pathlib.Path) -> None:
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required to save attention heatmaps.") from e
    Image.fromarray(arr).save(path)


# ── Attention reduction ─────────────────────────────────────────────────────


def split_attention_per_camera(
    attn: torch.Tensor,
    *,
    num_cameras: int,
    tokens_per_cam: int,
    key_offset: int = 0,
    query_slice: slice | None = None,
) -> torch.Tensor:
    """Reduce a 4-D attention tensor ``[B, H, Q, K]`` to per-camera 2-D heatmaps.

    Heads and queries are mean-reduced. Returns ``[B, num_cameras, side, side]``
    where ``side = isqrt(tokens_per_cam)``.
    """
    if attn.ndim != 4:
        raise ValueError(f"expected attn of shape [B, H, Q, K], got {tuple(attn.shape)}")
    if query_slice is not None:
        attn = attn[:, :, query_slice, :]
    attn = attn.mean(dim=1).mean(dim=1)  # [B, K]

    side = int(math.isqrt(tokens_per_cam))
    if side * side != tokens_per_cam:
        raise ValueError(f"tokens_per_cam={tokens_per_cam} is not a perfect square")

    cams = []
    for i in range(num_cameras):
        start = key_offset + i * tokens_per_cam
        end = start + tokens_per_cam
        cam_attn = attn[..., start:end].reshape(*attn.shape[:-1], side, side)
        cams.append(cam_attn)
    return torch.stack(cams, dim=1)


# ── Top-level save helper ───────────────────────────────────────────────────


def save_camera_heatmaps(
    images: Mapping[str, torch.Tensor],
    image_keys: Iterable[str],
    cam_attn_maps: torch.Tensor,
    save_dir: pathlib.Path | str,
    tag: str,
    *,
    alpha: float = 0.5,
    save_originals: bool = True,
) -> list[pathlib.Path]:
    """Save attention overlays for each batch element / camera.

    Args:
        images: mapping ``camera_key -> tensor[B, C, H, W]`` (or ``[B, H, W, C]``)
            in the model's pre-processed range (``[-1, 1]``).
        image_keys: ordered camera names matching the model's image order.
        cam_attn_maps: ``[B, num_cameras, side, side]`` from
            :func:`split_attention_per_camera`.
        save_dir: directory to write PNGs to (created if missing).
        tag: filename prefix, e.g. ``"layer10_step0"``.
        alpha: blending factor for the heatmap on the image.
        save_originals: also write the unmodified camera image alongside.
    """
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    image_keys = list(image_keys)

    written: list[pathlib.Path] = []
    bsize = cam_attn_maps.shape[0]
    for b in range(bsize):
        for cam_idx, key in enumerate(image_keys):
            if key not in images or cam_idx >= cam_attn_maps.shape[1]:
                continue
            img_uint8 = _to_uint8_image(images[key][b])
            hm = cam_attn_maps[b, cam_idx].detach().float().cpu().numpy()
            overlay = overlay_heatmap(img_uint8, hm, alpha=alpha)
            stem = f"{tag}_b{b}_{key}"
            overlay_path = save_dir / f"{stem}_overlay.png"
            _save_png(overlay, overlay_path)
            written.append(overlay_path)
            if save_originals:
                orig_path = save_dir / f"{stem}_orig.png"
                _save_png(img_uint8, orig_path)
                written.append(orig_path)
    return written
