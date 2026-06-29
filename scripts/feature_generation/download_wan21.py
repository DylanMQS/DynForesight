#!/usr/bin/env python3
"""Download Wan2.1 model weights from Hugging Face into the local repo layout
expected by ``precompute_wan21dit_features.py`` (and the rest of the
``Wan2.1`` pipeline).

By default this fetches **Wan2.1-T2V-1.3B** into

    src/videomodel/Wan2.1/Wan2.1-T2V-1.3B/

Honors the ``HF_ENDPOINT`` env var (e.g. ``https://hf-mirror.com``) and
``HF_HOME``, so it works transparently behind the China mirror that this
machine already has configured.

Behaviour:
* Uses ``huggingface_hub.snapshot_download`` — resumable, multi-threaded,
  and idempotent (re-running skips files whose hashes already match).
* By default downloads only the files actually needed for DiT feature
  extraction (VAE + DiT + T5 + tokenizer + LICENSE + README). Pass
  ``--all`` to grab everything in the repo (including ``assets/`` and
  ``examples/``).
* Files land in ``--local-dir`` directly (not symlinked), so the path you
  hand to ``--model-dir`` is self-contained and can be moved freely.

Examples:
    # Default — Wan2.1-T2V-1.3B into src/videomodel/Wan2.1/Wan2.1-T2V-1.3B/
    python scripts/download_wan21.py

    # T2V-14B variant
    python scripts/download_wan21.py --variant 14B

    # Custom destination + grab the whole repo (assets, examples, ...)
    python scripts/download_wan21.py --local-dir /mnt/shared/Wan2.1-T2V-1.3B --all

    # Specific repo override (e.g. VACE-1.3B)
    python scripts/download_wan21.py --repo-id Wan-AI/Wan2.1-VACE-1.3B \\
        --local-dir src/videomodel/Wan2.1/Wan2.1-VACE-1.3B
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Files that ``precompute_wan21dit_features.py`` actually opens.  Globs are
# passed straight through to ``snapshot_download(allow_patterns=...)``.
_ESSENTIAL_PATTERNS: List[str] = [
    "Wan2.1_VAE.pth",
    "config.json",
    "configuration.json",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model-*.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/*",
    "LICENSE.txt",
    "README.md",
]

_VARIANT_TO_REPO = {
    "1.3B": "Wan-AI/Wan2.1-T2V-1.3B",
    "14B": "Wan-AI/Wan2.1-T2V-14B",
    "i2v-14B-480P": "Wan-AI/Wan2.1-I2V-14B-480P",
    "i2v-14B-720P": "Wan-AI/Wan2.1-I2V-14B-720P",
    "flf2v-14B": "Wan-AI/Wan2.1-FLF2V-14B-720P",
    "vace-1.3B": "Wan-AI/Wan2.1-VACE-1.3B",
    "vace-14B": "Wan-AI/Wan2.1-VACE-14B",
}


def _default_local_dir(repo_id: str) -> str:
    """Mirror the repo basename under ``src/videomodel/Wan2.1/`` (relative to this script)."""
    base = os.path.basename(repo_id)  # e.g. "Wan2.1-T2V-1.3B"
    wan21_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "src", "videomodel", "Wan2.1",
    ))
    return os.path.join(wan21_dir, base)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variant",
        choices=list(_VARIANT_TO_REPO.keys()),
        default="1.3B",
        help="Shortcut for the Wan2.1 model variant (default: 1.3B → Wan-AI/Wan2.1-T2V-1.3B). "
             "Ignored if --repo-id is also given.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Explicit Hugging Face repo id (overrides --variant). "
             "Example: Wan-AI/Wan2.1-T2V-1.3B",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Destination directory (default: <pi05>/src/videomodel/Wan2.1/<basename>). "
             "Existing files with matching SHA are skipped; partial downloads resume.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download everything in the repo (including assets/ and examples/). "
             "By default only the files needed for DiT feature extraction are fetched.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent download threads (default: 8). Lower if your network is flaky.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision / commit / tag to pin (default: latest main).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (rarely needed — Wan2.1 is public).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually fetching.",
    )
    args = parser.parse_args()

    repo_id = args.repo_id or _VARIANT_TO_REPO[args.variant]
    local_dir = args.local_dir or _default_local_dir(repo_id)
    local_dir = os.path.abspath(local_dir)

    allow_patterns: Optional[List[str]] = None if args.all else _ESSENTIAL_PATTERNS

    hf_endpoint = os.environ.get("HF_ENDPOINT", "(default huggingface.co)")
    hf_home = os.environ.get("HF_HOME", "(default ~/.cache/huggingface)")

    logging.info(f"Repo            : {repo_id}")
    logging.info(f"Destination     : {local_dir}")
    logging.info(f"HF_ENDPOINT     : {hf_endpoint}")
    logging.info(f"HF_HOME         : {hf_home}")
    logging.info(f"Filter          : {'ALL files' if args.all else f'{len(allow_patterns)} essential patterns'}")
    if not args.all:
        for p in allow_patterns:
            logging.info(f"  - {p}")
    logging.info(f"Max workers     : {args.max_workers}")
    if args.revision:
        logging.info(f"Revision        : {args.revision}")

    if args.dry_run:
        logging.info("Dry run — exiting without downloading.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        logging.error("huggingface_hub is not installed. Run `pip install \"huggingface_hub[cli]\"` first.")
        raise SystemExit(1) from e

    os.makedirs(local_dir, exist_ok=True)

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        max_workers=args.max_workers,
        revision=args.revision,
        token=args.token,
        # local_dir_use_symlinks defaults to "auto" in modern hf_hub which on
        # local_dir falls back to copying — that's exactly what we want, so the
        # destination directory is self-contained.
    )

    logging.info(f"Done. Files are at: {path}")
    logging.info("")
    logging.info("Next step:")
    logging.info(
        f"  python scripts/precompute_wan21dit_features.py "
        f"--config-name <your_pi05_video_config> --model-dir {path} "
        f"--feat-block-indices 19"
    )

    # Sanity check: ensure all four critical files actually landed (only when
    # filtering — for --all the user is on their own).
    if not args.all:
        critical = [
            "Wan2.1_VAE.pth",
            "config.json",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "google/umt5-xxl/spiece.model",
        ]
        missing = [f for f in critical if not os.path.exists(os.path.join(path, f))]
        # The DiT can be either single-file or sharded; accept either layout.
        has_dit = (
            os.path.exists(os.path.join(path, "diffusion_pytorch_model.safetensors"))
            or os.path.exists(os.path.join(path, "diffusion_pytorch_model.safetensors.index.json"))
        )
        if not has_dit:
            missing.append("diffusion_pytorch_model.safetensors[.index.json]")
        if missing:
            logging.warning(
                f"The following expected files are missing under {path}: {missing}. "
                f"You may want to re-run with --all to fetch the full repo."
            )


if __name__ == "__main__":
    main()
