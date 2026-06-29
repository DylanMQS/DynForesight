#!/usr/bin/env python3
"""Standalone shard-to-mmap merger for WanDiT / Cosmos-DiT feature cache folders.

Run this on ANY server that has access to the shared directory to merge one
block's shard folder into a single .mmap + .meta.pt file.  Multiple instances
can run in parallel — one per block folder — across different machines.

Supported folder name patterns (auto-derived output paths):

    WanDiT  :  ``_wandit_tmp__{keys}__{frames}__blk{N}_t{T}``
               →  ``backup/wandit_cache__{keys}__{frames}__blk{N}_t{T}``
               (output placed in a ``backup/`` subdir next to the tmp folder)

    Wan21DiT:  ``_wan21dit_tmp__{keys}__{frames}__blk{N}_t{T}``
               →  ``backup/wan21dit_cache__{keys}__{frames}__blk{N}_t{T}``
               (Wan2.1-specific counterpart of the WanDiT layout)

    Cosmos  :  ``_cosmosdit_{rev}_tmp__{keys}__{frames}__blk{N}_t{T}``
               →  ``cosmosdit_{rev}_cache__{keys}__{frames}__blk{N}_t{T}``
               (output placed directly next to the tmp folder, matching
               ``precompute_cosmos_dit_features.py``'s ``_out_prefix_for_block``)

Anything else falls back to ``<basename>__merged`` next to the tmp folder.

Usage:
    python scripts/merge_wandit_shards.py assets/pi05_libero_video_align/lerobot_all_repo/_wandit_tmp__top+cam__f0_1_2__blk3_t300

    # Cosmos shards (works without any extra flags):
    python scripts/merge_wandit_shards.py assets/pi05_aloha_video_align/lerobot_all_repo/_cosmosdit_post_tmp__top+cam__f0_1_2__blk3_t0300

    # Keep the shard folder after merging (for debugging):
    python scripts/merge_wandit_shards.py --keep-shards /path/to/tmp_folder

    # Explicit output prefix (overrides auto-detection):
    python scripts/merge_wandit_shards.py --output-prefix /path/to/cache_prefix /path/to/tmp_folder

Shard files are deleted by default after successful merge; use ``--keep-shards``
to preserve them.
"""

import argparse
import glob
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import torch
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ``_cosmosdit_{rev}_tmp__...`` where ``{rev}`` is a short tag like ``post`` /
# ``pre`` / ``distill`` (auto-derived from the HF revision in the producer).
_COSMOS_TMP_RE = re.compile(r"^_cosmosdit_(?P<rev>[A-Za-z0-9_\-]+)_tmp__(?P<rest>.+)$")


def _derive_output_prefix(tmp_dir: str) -> str:
    """Derive the mmap output prefix from the tmp folder path.

    Handles both WanDiT and Cosmos-DiT layouts (see module docstring).
    """
    parent = os.path.dirname(os.path.abspath(tmp_dir))
    basename = os.path.basename(os.path.normpath(tmp_dir))

    if basename.startswith("_wandit_tmp__"):
        backup_dir = os.path.join(parent, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        out_base = "wandit_cache__" + basename[len("_wandit_tmp__"):]
        return os.path.join(backup_dir, out_base)

    if basename.startswith("_wan21dit_tmp__"):
        backup_dir = os.path.join(parent, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        out_base = "wan21dit_cache__" + basename[len("_wan21dit_tmp__"):]
        return os.path.join(backup_dir, out_base)

    cosmos_m = _COSMOS_TMP_RE.match(basename)
    if cosmos_m is not None:
        rev = cosmos_m.group("rev")
        rest = cosmos_m.group("rest")
        out_base = f"cosmosdit_{rev}_cache__{rest}"
        return os.path.join(parent, out_base)

    out_base = basename + "__merged"
    return os.path.join(parent, out_base)


def _scan_shard(sf: str) -> Tuple[str, List[int], List[str], Tuple[int, ...]]:
    """Load one shard and return its keys + per-sample structure.

    Returns ``(shard_path, keys, cam_names_sorted, sample_shape)``.  Tensor
    data is dropped before returning (via ``del shard``) so peak memory across
    workers is bounded by *num_workers* shards, not all of them.
    """
    shard = torch.load(sf, map_location="cpu", weights_only=False)
    keys = list(shard.keys())
    cam_names: List[str] = []
    sample_shape: Tuple[int, ...] = ()
    if keys:
        any_v = shard[keys[0]]
        cam_names = sorted(any_v.keys())
        sample_shape = tuple(any_v[cam_names[0]].shape)
    del shard
    return sf, keys, cam_names, sample_shape


def _write_shard_to_mmap(
    sf: str,
    fp: np.memmap,
    key_to_pos: dict,
    cam_names: List[str],
    np_dtype: "np.typing.DTypeLike",
) -> int:
    """Load one shard and copy its tensors into the right slots of ``fp``.

    Multiple workers can call this concurrently because the producer guarantees
    each shard's sample indices are disjoint (rank-strided), so the destination
    slices in the mmap never overlap.
    """
    shard = torch.load(sf, map_location="cpu", weights_only=False)
    n_written = 0
    for k, per_cam in shard.items():
        pos = key_to_pos[k]
        for j, c in enumerate(cam_names):
            t = per_cam[c]
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float16)
            fp[pos, j] = t.numpy().astype(np_dtype, copy=False)
        n_written += 1
    del shard
    return n_written


def shards_to_mmap(tmp_dir: str, output_prefix: str, num_workers: int = 8):
    """Stream-convert per-block shard .pt files to a single mmap file.

    Shard format: ``{sample_idx: {cam_name: tensor}}``.

    Produces:
        ``<output_prefix>.mmap``      — shape ``(N, n_cams, *feat_shape)``
        ``<output_prefix>.meta.pt``   — ``{shape, dtype, key_to_pos, cam_names}``

    Both the scan and the write passes are parallelised with a thread pool of
    ``num_workers`` threads.  Peak memory ≈ ``num_workers`` shards loaded
    concurrently.  Set ``num_workers=1`` for the original sequential behaviour.
    """
    shard_files = sorted(glob.glob(os.path.join(tmp_dir, "rank*_chunk*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"No shards found in {tmp_dir}")

    num_workers = max(1, int(num_workers))
    logging.info(
        f"Found {len(shard_files)} shard files in {tmp_dir}; "
        f"scanning keys & shapes with {num_workers} threads ..."
    )

    all_keys: List[int] = []
    sample_shape: Optional[Tuple[int, ...]] = None
    cam_names: Optional[List[str]] = None

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_scan_shard, sf) for sf in shard_files]
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Scanning shards"):
            _sf, keys, cams, shape = fut.result()
            all_keys.extend(keys)
            if sample_shape is None and cams:
                cam_names = cams
                sample_shape = shape

    if sample_shape is None or cam_names is None:
        raise RuntimeError(f"All shards in {tmp_dir} appear to be empty")

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
    logging.info(f"Writing to mmap with {num_workers} threads ...")
    total_written = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(_write_shard_to_mmap, sf, fp, key_to_pos, cam_names, np_dtype)
            for sf in shard_files
        ]
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Writing to mmap"):
            total_written += fut.result()

    fp.flush()
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
    logging.info(
        f"Done: {n} unique samples (wrote {total_written} entries) "
        f"→ {mmap_path} ({size_gb:.2f} GB) + {meta_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Merge WanDiT / Cosmos-DiT shard .pt files into a single .mmap file. "
                    "Run one instance per block folder — parallelise across servers.",
    )
    parser.add_argument(
        "tmp_dir",
        help="Path to the _wandit_tmp__* or _cosmosdit_*_tmp__* shard folder to merge",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Override the auto-derived output prefix. "
             "Default: WanDiT → backup/wandit_cache__... ; "
             "Cosmos → cosmosdit_{rev}_cache__... next to the tmp folder.",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep the shard folder after merging (default: delete it)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of threads used for parallel shard load + mmap write "
             "(default: 8). Set to 1 for the original sequential behaviour. "
             "Memory grows roughly linearly with this value (one shard per worker).",
    )
    args = parser.parse_args()

    tmp_dir = os.path.abspath(args.tmp_dir)
    if not os.path.isdir(tmp_dir):
        raise FileNotFoundError(f"Directory not found: {tmp_dir}")

    output_prefix = args.output_prefix or _derive_output_prefix(tmp_dir)
    logging.info(f"Input  folder : {tmp_dir}")
    logging.info(f"Output prefix : {output_prefix}")
    logging.info(f"Num workers   : {args.num_workers}")

    shards_to_mmap(tmp_dir, output_prefix, num_workers=args.num_workers)

    if not args.keep_shards:
        logging.info(f"Cleaning up {tmp_dir} ...")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logging.info("Cleanup done.")
    else:
        logging.info(f"Shard folder kept: {tmp_dir}")


if __name__ == "__main__":
    main()
