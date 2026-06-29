"""Extract every frame from each video in a directory.

For each video file under ``--input-dir`` (non-recursive by default), this
script creates a sibling folder named after the video stem under
``--output-dir`` and writes every decoded frame as an image inside it.

Example:
    python scripts/extract_video_frames.py \
        --input-dir data/libero/videos/libero_10 \
        --output-dir data/libero/videos/libero_10_frames

Output layout::

    libero_10_frames/
        <video_stem>/
            frame_000000.jpg
            frame_000001.jpg
            ...
            meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_frames(
    video_path: Path,
    out_dir: Path,
    image_ext: str = "jpg",
    jpeg_quality: int = 95,
    png_compression: int = 3,
    overwrite: bool = False,
) -> dict:
    """Decode ``video_path`` and dump every frame into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"

    if meta_path.exists() and not overwrite:
        try:
            existing = json.loads(meta_path.read_text())
            if existing.get("status") == "ok":
                return {
                    "video": str(video_path),
                    "out_dir": str(out_dir),
                    "skipped": True,
                    **existing,
                }
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    declared_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if image_ext.lower() in {"jpg", "jpeg"}:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        suffix = ".jpg"
    elif image_ext.lower() == "png":
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]
        suffix = ".png"
    else:
        raise ValueError(f"Unsupported image_ext: {image_ext}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            out_path = out_dir / f"frame_{frame_idx:06d}{suffix}"
            if not cv2.imwrite(str(out_path), frame, encode_params):
                raise RuntimeError(f"Failed to write {out_path}")
            frame_idx += 1
    finally:
        cap.release()

    meta = {
        "status": "ok",
        "video": str(video_path),
        "out_dir": str(out_dir),
        "fps": fps,
        "width": width,
        "height": height,
        "declared_frame_count": declared_count,
        "extracted_frame_count": frame_idx,
        "image_ext": suffix.lstrip("."),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {**meta, "skipped": False}


def _worker(args):
    (video_path, out_dir, image_ext, jpeg_quality, png_compression, overwrite) = args
    try:
        return extract_frames(
            Path(video_path),
            Path(out_dir),
            image_ext=image_ext,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
            overwrite=overwrite,
        )
    except Exception as e:  # propagate as result for logging
        return {
            "status": "error",
            "video": str(video_path),
            "out_dir": str(out_dir),
            "error": repr(e),
        }


def find_videos(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/mnt/workspace/mqs/workspace/VLA/openpi/data/libero/videos/libero_10"),
        help="Directory containing video files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write per-video frame folders. Defaults to '<input-dir>_frames'.",
    )
    parser.add_argument("--ext", choices=["jpg", "png"], default="jpg", help="Output image format.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel processes (one per video).")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if meta.json says done.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        print(f"[error] input dir does not exist: {input_dir}", file=sys.stderr)
        return 2

    output_dir: Path = args.output_dir or input_dir.parent / f"{input_dir.name}_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir, args.recursive)
    if not videos:
        print(f"[warn] no videos found under {input_dir}", file=sys.stderr)
        return 0

    print(f"Found {len(videos)} videos in {input_dir}")
    print(f"Writing frames into {output_dir}")

    tasks = [
        (
            str(v),
            str(output_dir / v.stem),
            args.ext,
            args.jpeg_quality,
            args.png_compression,
            args.overwrite,
        )
        for v in videos
    ]

    n_ok = n_skip = n_err = 0
    if args.num_workers <= 1:
        for t in tasks:
            res = _worker(t)
            n_ok, n_skip, n_err = _report(res, n_ok, n_skip, n_err)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_worker, t) for t in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                n_ok, n_skip, n_err = _report(res, n_ok, n_skip, n_err)

    print(f"\nDone. ok={n_ok} skipped={n_skip} errors={n_err} (total={len(videos)})")
    return 0 if n_err == 0 else 1


def _report(res: dict, n_ok: int, n_skip: int, n_err: int) -> tuple[int, int, int]:
    status = res.get("status")
    video = res.get("video", "?")
    if status == "ok" and res.get("skipped"):
        n_skip += 1
        print(f"[skip] {video} -> {res.get('extracted_frame_count')} frames already extracted")
    elif status == "ok":
        n_ok += 1
        print(
            f"[ok]   {video} -> {res.get('extracted_frame_count')} frames "
            f"({res.get('width')}x{res.get('height')} @ {res.get('fps'):.2f} fps)"
        )
    else:
        n_err += 1
        print(f"[err]  {video}: {res.get('error')}", file=sys.stderr)
    return n_ok, n_skip, n_err


if __name__ == "__main__":
    raise SystemExit(main())
