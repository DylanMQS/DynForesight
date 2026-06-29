"""Recover the attention heatmap from an (original, overlay) PNG pair and
re-apply it to a different image.

The original overlay was produced by
:func:`openpi.models_pytorch.attention_viz.overlay_heatmap` with the formula

    overlay = (1 - alpha) * orig + alpha * jet(h)

where ``h ∈ [0, 1]`` is the per-pixel min-max-normalized heatmap. With a flat
alpha the equation is closed-form per pixel, but to keep things robust against
JPEG-style quantisation we still pick the ``h`` that minimises the per-pixel
reconstruction error over a dense grid (vectorised; runs in well under a
second on a 224x224 PNG).

Usage
-----
    python openpi/scripts/transfer_heatmap.py \
        --orig    path/to/step0030_..._orig.png \
        --overlay path/to/step0030_..._overlay.png \
        --target  path/to/another_image.png \
        --output  transferred.png

Optional flags (must match the value used when the overlay was generated):

    --alpha            0.5
    --n-candidates     256 (search resolution)

You can also provide ``--save-heatmap recovered.png`` to dump the recovered
heatmap as a standalone (jet-coloured) image.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

# Make sure ``openpi`` is importable when the script is run from the repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from openpi.models_pytorch.attention_viz import _jet_colormap  # noqa: E402


# ── recovery ────────────────────────────────────────────────────────────────


def recover_heatmap(
    orig: np.ndarray,
    overlay: np.ndarray,
    *,
    alpha: float = 0.5,
    n_candidates: int = 256,
) -> np.ndarray:
    """Recover the per-pixel normalized heatmap. Returns a ``[H, W]`` ``float32`` in ``[0, 1]``."""
    if orig.shape != overlay.shape:
        raise ValueError(f"orig {orig.shape} and overlay {overlay.shape} differ")
    if orig.ndim != 3 or orig.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {orig.shape}")

    h_grid = np.linspace(0.0, 1.0, n_candidates).astype(np.float32)
    colors = _jet_colormap(h_grid).astype(np.float32)  # (N, 3)

    H, W = orig.shape[:2]
    orig_f = orig.astype(np.float32)
    over_f = overlay.astype(np.float32)

    best_err = np.full((H, W), np.inf, dtype=np.float32)
    best_h = np.zeros((H, W), dtype=np.float32)

    for i in range(n_candidates):
        c = colors[i]
        pred = orig_f * (1.0 - alpha) + c[None, None, :] * alpha
        err = np.sum((pred - over_f) ** 2, axis=-1)
        improved = err < best_err
        best_err[improved] = err[improved]
        best_h[improved] = h_grid[i]

    return best_h


# ── re-apply to a new image ─────────────────────────────────────────────────


def _resize_2d(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if arr.shape == (h, w):
        return arr
    img = Image.fromarray((arr.clip(0.0, 1.0) * 65535).astype(np.uint16), mode="I;16")
    img = img.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 65535.0


def apply_heatmap(
    target: np.ndarray,
    h_normalized: np.ndarray,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    """Re-blend a recovered heatmap onto ``target`` (uint8 RGB) with a flat alpha."""
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB target, got {target.shape}")

    h = _resize_2d(h_normalized.astype(np.float32), target.shape[:2])
    color = _jet_colormap(h).astype(np.float32)
    img = target.astype(np.float32)
    blended = img * (1.0 - alpha) + color * alpha
    return blended.clip(0, 255).astype(np.uint8)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orig", required=True, help="path to *_orig.png (the image the heatmap was overlaid on)")
    p.add_argument("--overlay", required=True, help="path to *_overlay.png (heatmap-blended image)")
    p.add_argument("--target", required=True, help="path to the new image to receive the heatmap")
    p.add_argument("--output", required=True, help="output PNG path")
    p.add_argument("--save-heatmap", default=None, help="optional path to also save the recovered heatmap as a standalone PNG")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--n-candidates", type=int, default=256, help="search resolution for h ∈ [0,1]")
    p.add_argument("--target-alpha", type=float, default=None, help="alpha for re-blending (defaults to --alpha)")
    args = p.parse_args()

    orig = np.asarray(Image.open(args.orig).convert("RGB"))
    overlay = np.asarray(Image.open(args.overlay).convert("RGB"))
    target = np.asarray(Image.open(args.target).convert("RGB"))

    print(f"Recovering heatmap from {orig.shape[:2]} ...")
    h = recover_heatmap(orig, overlay, alpha=args.alpha, n_candidates=args.n_candidates)
    print(f"  recovered heatmap stats: min={h.min():.3f} max={h.max():.3f} mean={h.mean():.3f}")

    if args.save_heatmap:
        col = _jet_colormap(h).astype(np.uint8)
        Image.fromarray(col).save(args.save_heatmap)
        print(f"  wrote standalone heatmap: {args.save_heatmap}")

    out = apply_heatmap(
        target, h,
        alpha=args.alpha if args.target_alpha is None else args.target_alpha,
    )
    Image.fromarray(out).save(args.output)
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
