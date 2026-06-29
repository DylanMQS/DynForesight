#!/usr/bin/env python3
"""Convert a dict-based .pt feature cache to mmap-friendly flat format.

The original cache format is::

    {sample_idx: {cam_name: Tensor[C, T, H, W], ...}, ...}

This script converts it to two files:

- ``<name>.mmap``  — a flat binary file of shape ``(N, n_cams, C, T, H, W)`` in float16
- ``<name>.meta.pt`` — a small metadata dict with shape/dtype/key mapping

Usage::

    python scripts/convert_cache_to_mmap.py \
        --input  assets/.../vae_cache__image+wrist_image__f0_1_2_3_4_5_6_7_8.pt \
        --output assets/.../vae_cache__image+wrist_image__f0_1_2_3_4_5_6_7_8

    # Produces:
    #   *.mmap      (flat binary)
    #   *.meta.pt   (small metadata)

For very large caches (hundreds of GB) the script streams shards to avoid OOM.
"""

import argparse
import glob
import logging
import os

import numpy as np
import torch
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def convert_single_pt(input_path: str, output_prefix: str):
    """Convert a single .pt dict cache to mmap format."""
    logging.info(f"Loading cache from {input_path} ...")
    cache = torch.load(input_path, map_location="cpu", weights_only=False)
    _convert_dict(cache, output_prefix)


def convert_sharded(shard_dir: str, output_prefix: str):
    """Convert a directory of rank*_chunk*.pt shards directly (avoids merging)."""
    pattern = os.path.join(shard_dir, "rank*_chunk*.pt")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(f"No shards found in {shard_dir}")

    logging.info(f"Found {len(shard_files)} shard files in {shard_dir}")

    # First pass: collect all keys and determine shape
    logging.info("Pass 1: scanning keys and shapes ...")
    all_keys = []
    sample_shape = None
    cam_names = None
    dtype = None
    for sf in tqdm.tqdm(shard_files, desc="Scanning shards"):
        shard = torch.load(sf, map_location="cpu", weights_only=False)
        for k, v in shard.items():
            all_keys.append(k)
            if sample_shape is None:
                cam_names = list(v.keys())  # preserve insertion order
                sample_shape = v[cam_names[0]].shape
                dtype = v[cam_names[0]].dtype
        del shard

    all_keys = sorted(set(all_keys))
    n = len(all_keys)
    n_cams = len(cam_names)
    np_dtype = _torch_to_numpy_dtype(dtype)
    logging.info(f"  {n} samples, {n_cams} cameras, shape={sample_shape}, dtype={dtype}")

    # Create mmap file
    mmap_path = output_prefix + ".mmap"
    full_shape = (n, n_cams, *sample_shape)
    logging.info(f"Creating mmap file {mmap_path} with shape {full_shape} ...")
    fp = np.memmap(mmap_path, dtype=np_dtype, mode="w+", shape=full_shape)

    key_to_pos = {k: i for i, k in enumerate(all_keys)}

    # Second pass: fill data
    logging.info("Pass 2: writing data to mmap ...")
    for sf in tqdm.tqdm(shard_files, desc="Writing to mmap"):
        shard = torch.load(sf, map_location="cpu", weights_only=False)
        for k, per_cam in shard.items():
            pos = key_to_pos[k]
            for j, c in enumerate(cam_names):
                fp[pos, j] = _to_numpy(per_cam[c], np_dtype)
        del shard
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
    logging.info(f"Saved metadata to {meta_path}")
    size_gb = os.path.getsize(mmap_path) / 1e9
    logging.info(f"Done. mmap size: {size_gb:.2f} GB")


def _convert_dict(cache: dict, output_prefix: str):
    keys = sorted(cache.keys())
    sample_val = cache[keys[0]]
    cam_names = list(sample_val.keys())  # preserve insertion order to match LookupVaeCache
    sample_shape = sample_val[cam_names[0]].shape
    dtype = sample_val[cam_names[0]].dtype
    np_dtype = _torch_to_numpy_dtype(dtype)

    n = len(keys)
    n_cams = len(cam_names)
    full_shape = (n, n_cams, *sample_shape)

    mmap_path = output_prefix + ".mmap"
    logging.info(f"Creating mmap {mmap_path}: shape={full_shape}, dtype={np_dtype}")
    fp = np.memmap(mmap_path, dtype=np_dtype, mode="w+", shape=full_shape)

    key_to_pos = {}
    for i, k in enumerate(tqdm.tqdm(keys, desc="Writing to mmap")):
        key_to_pos[k] = i
        per_cam = cache[k]
        for j, c in enumerate(cam_names):
            fp[i, j] = _to_numpy(per_cam[c], np_dtype)
        if (i + 1) % 10000 == 0:
            fp.flush()

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
    logging.info(f"Saved metadata ({len(key_to_pos)} keys) to {meta_path}")
    size_gb = os.path.getsize(mmap_path) / 1e9
    logging.info(f"Done. mmap size: {size_gb:.2f} GB")


def _torch_to_numpy_dtype(dtype):
    return {
        torch.float16: np.float16,
        torch.bfloat16: np.float16,  # numpy has no bfloat16, store as float16
        torch.float32: np.float32,
    }.get(dtype, np.float16)


def _to_numpy(tensor: torch.Tensor, target_np_dtype) -> np.ndarray:
    """Convert a torch tensor to numpy, handling bfloat16 which numpy doesn't support."""
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float16)
    return tensor.numpy().astype(target_np_dtype, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Convert .pt feature cache to mmap format")
    parser.add_argument("--input", required=True,
                        help="Path to .pt cache file, or directory of rank*_chunk*.pt shards")
    parser.add_argument("--output", required=True,
                        help="Output prefix (will produce <prefix>.mmap and <prefix>.meta.pt)")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        convert_sharded(args.input, args.output)
    else:
        convert_single_pt(args.input, args.output)


if __name__ == "__main__":
    main()
