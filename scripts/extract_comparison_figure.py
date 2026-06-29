#!/usr/bin/env python3
"""Extract paper-ready figures from GT-vs-Pred comparison videos.

The comparison videos produced by ``visualize_video_projector.py`` (with
``--save-comparison --comparison-layout gt_pred``) have the layout::

    ┌──────────────┬──────────────┐
    │  GT label    │  pred label  │   ← 8 px coloured label bar
    ├──────────────┼──────────────┤
    │              │              │
    │  GT  (224²)  │ pred (224²)  │   ← actual frames
    │              │              │
    └──────────────┴──────────────┘

This script samples ``--num-frames`` frames uniformly across each video and
composes a single PNG per video with two rows:

    Row 1: GT     (frame_0  frame_1  ...  frame_{N-1})
    Row 2: Pred   (frame_0  frame_1  ...  frame_{N-1})

Optionally, row labels and frame indices are drawn for clarity.

Usage::

    python scripts/extract_comparison_figure.py \
        --input-dir  openpi/viz_outputs/viz_select \
        --output-dir openpi/viz_outputs/viz_select/figures \
        --num-frames 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABEL_BAR_PX = 8


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for p in candidates:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _read_video_uint8(path: Path) -> np.ndarray:
    """Return an array of shape [T, H, W, 3] in uint8.

    Tries multiple backends so the script works across mismatched
    ``imageio``/``pyav``/``cv2`` versions found in different venvs.
    """
    last_err: Exception | None = None
    for plugin in ("FFMPEG", "pyav"):
        try:
            frames = iio.imread(str(path), plugin=plugin)
            if frames.ndim == 4 and frames.shape[-1] == 3:
                return frames.astype(np.uint8, copy=False)
        except Exception as e:
            last_err = e
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture could not open {path}")
        out = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not out:
            raise RuntimeError(f"cv2 returned 0 frames for {path}")
        return np.stack(out, axis=0).astype(np.uint8, copy=False)
    except Exception as e:
        last_err = e
    raise RuntimeError(f"Could not decode {path} via any backend: {last_err!r}")


def _strip_and_split(video: np.ndarray, label_bar_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove the top label bar and split the frame into (gt, pred)."""
    if video.shape[1] <= label_bar_px:
        raise ValueError(
            f"Video height {video.shape[1]} is not larger than label_bar_px={label_bar_px}; "
            "cannot strip the label bar."
        )
    content = video[:, label_bar_px:, :, :]
    W = content.shape[2]
    if W % 2 != 0:
        raise ValueError(f"Content width {W} is not divisible by 2 (expected GT|pred halves).")
    half = W // 2
    gt = content[:, :, :half, :]
    pred = content[:, :, half:, :]
    return gt, pred


def _pick_frame_indices(num_total: int, num_keep: int) -> list[int]:
    if num_keep <= 0:
        raise ValueError("num-frames must be >= 1")
    if num_keep == 1:
        return [num_total // 2]
    if num_keep >= num_total:
        return list(range(num_total))
    # Always include first and last; uniformly space the rest.
    return [int(round(i * (num_total - 1) / (num_keep - 1))) for i in range(num_keep)]


def _compose_figure(
    gt_frames: np.ndarray,         # [N, H, W, 3]
    pred_frames: np.ndarray,       # [N, H, W, 3]
    *,
    inner_gap: int,
    row_gap: int,
    margin: int,
    label_width: int,
    show_labels: bool,
    show_frame_indices: bool,
    frame_indices: list[int],
    title: str | None,
) -> Image.Image:
    N, H, W, _ = gt_frames.shape
    assert pred_frames.shape == gt_frames.shape

    title_h = 0
    if title:
        title_h = max(20, H // 10)

    frame_idx_h = 0
    if show_frame_indices:
        frame_idx_h = max(16, H // 14)

    grid_w = N * W + (N - 1) * inner_gap
    grid_h = 2 * H + row_gap

    total_w = margin * 2 + (label_width if show_labels else 0) + grid_w
    total_h = margin * 2 + title_h + frame_idx_h + grid_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if title:
        font = _load_font(max(14, H // 12))
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            ((total_w - tw) // 2, margin + (title_h - th) // 2 - bbox[1]),
            title,
            fill=(0, 0, 0),
            font=font,
        )

    grid_x0 = margin + (label_width if show_labels else 0)
    grid_y0 = margin + title_h + frame_idx_h

    if show_frame_indices:
        font = _load_font(max(12, H // 18))
        for j, fi in enumerate(frame_indices):
            text = f"t={fi}"
            x = grid_x0 + j * (W + inner_gap)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x + (W - tw) // 2, grid_y0 - frame_idx_h + (frame_idx_h - th) // 2 - bbox[1]),
                text,
                fill=(50, 50, 50),
                font=font,
            )

    if show_labels:
        font = _load_font(max(16, H // 10))
        for row, (label, colour) in enumerate(
            [("GT", (40, 140, 60)), ("Pred", (200, 100, 40))]
        ):
            y_center = grid_y0 + row * (H + row_gap) + H // 2
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (margin + (label_width - tw) // 2,
                 y_center - th // 2 - bbox[1]),
                label,
                fill=colour,
                font=font,
            )

    for j in range(N):
        x = grid_x0 + j * (W + inner_gap)
        canvas.paste(Image.fromarray(gt_frames[j]),   (x, grid_y0))
        canvas.paste(Image.fromarray(pred_frames[j]), (x, grid_y0 + H + row_gap))

    return canvas


def _short_title_from_filename(name: str) -> str:
    """``ep1400_f0076of0140__pick_up_the_black_bowl..._image`` →
    ``ep1400 · f76/140 · pick up the black bowl ...``."""
    stem = Path(name).stem
    parts = stem.split("__")
    head = parts[0]
    prompt = " ".join(parts[1:-1]).replace("_", " ").strip() if len(parts) > 2 else ""
    if not prompt and len(parts) == 2:
        prompt = parts[1].replace("_", " ").strip()
    try:
        ep_part, frame_part = head.split("_", 1)
        ep_num = int(ep_part.replace("ep", ""))
        f, total = frame_part.replace("f", "").split("of")
        head_pretty = f"ep{ep_num} · f{int(f)}/{int(total)}"
    except Exception:
        head_pretty = head
    return f"{head_pretty}  ·  {prompt}" if prompt else head_pretty


def process_one(
    video_path: Path,
    out_path: Path,
    *,
    num_frames: int,
    label_bar_px: int,
    inner_gap: int,
    row_gap: int,
    margin: int,
    label_width: int,
    show_labels: bool,
    show_frame_indices: bool,
    show_title: bool,
) -> None:
    video = _read_video_uint8(video_path)
    gt, pred = _strip_and_split(video, label_bar_px)

    T = gt.shape[0]
    idxs = _pick_frame_indices(T, num_frames)
    gt_pick = gt[idxs]
    pred_pick = pred[idxs]

    title = _short_title_from_filename(video_path.name) if show_title else None
    fig = _compose_figure(
        gt_pick,
        pred_pick,
        inner_gap=inner_gap,
        row_gap=row_gap,
        margin=margin,
        label_width=label_width if show_labels else 0,
        show_labels=show_labels,
        show_frame_indices=show_frame_indices,
        frame_indices=idxs,
        title=title,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out_path, format="PNG", optimize=True)
    logging.info(
        f"[{video_path.name}]  T={T}  picked={idxs}  →  {out_path}  ({fig.size[0]}×{fig.size[1]})"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("openpi/viz_outputs/viz_select"),
        help="Folder containing comparison .mp4 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the resulting PNGs. "
             "Defaults to <input-dir>/figures.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="Glob pattern for input videos.",
    )
    parser.add_argument("--num-frames", type=int, default=5,
                        help="Number of frames sampled per video (default 5).")
    parser.add_argument("--label-bar-px", type=int, default=LABEL_BAR_PX,
                        help="Height of the coloured label bar at the top of each video frame.")
    parser.add_argument("--inner-gap", type=int, default=4,
                        help="Pixels between consecutive frames in a row.")
    parser.add_argument("--row-gap", type=int, default=6,
                        help="Pixels between the GT row and the Pred row.")
    parser.add_argument("--margin", type=int, default=12,
                        help="Outer margin in pixels.")
    parser.add_argument("--label-width", type=int, default=70,
                        help="Width of the per-row label column on the left (when --show-labels).")
    parser.add_argument("--no-labels", action="store_true",
                        help="Do not draw the GT / Pred row labels.")
    parser.add_argument("--no-frame-indices", action="store_true",
                        help="Do not draw t=k frame indices above each column.")
    parser.add_argument("--no-title", action="store_true",
                        help="Do not draw the per-video title at the top.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input dir does not exist: {input_dir}")

    output_dir = (args.output_dir or (input_dir / "figures")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob(args.pattern))
    if not videos:
        raise SystemExit(f"No videos matched {input_dir}/{args.pattern}")

    logging.info(f"Found {len(videos)} videos in {input_dir}")
    logging.info(f"Output: {output_dir}")

    for vp in videos:
        out = output_dir / (vp.stem + ".png")
        try:
            process_one(
                vp,
                out,
                num_frames=args.num_frames,
                label_bar_px=args.label_bar_px,
                inner_gap=args.inner_gap,
                row_gap=args.row_gap,
                margin=args.margin,
                label_width=args.label_width,
                show_labels=not args.no_labels,
                show_frame_indices=not args.no_frame_indices,
                show_title=not args.no_title,
            )
        except Exception as e:
            logging.exception(f"Failed on {vp}: {e}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
