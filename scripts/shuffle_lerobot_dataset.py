"""
Shuffle a LeRobot v2.0 dataset and create few-shot subsets.

Creates:
  - lerobot_all_repo_shuffled/        (100% episodes, shuffled order)
  - lerobot_all_repo_shuffled_25pct/  (25% episodes from shuffled order)
  - lerobot_all_repo_shuffled_5pct/   (5%)
  - lerobot_all_repo_shuffled_1pct/   (1%)

Usage:
    python shuffle_lerobot_dataset.py [--seed 42] [--src /path/to/lerobot_all_repo]
"""

import argparse
import copy
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CHUNK_SIZE = 1000
SEED = 42
SUBSETS = [
    ("shuffled", 1.0),
    ("shuffled_25pct", 0.25),
    ("shuffled_5pct", 0.05),
    ("shuffled_1pct", 0.01),
]


def read_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path, items):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def episode_parquet_path(base_dir, episode_index):
    chunk = episode_index // CHUNK_SIZE
    return os.path.join(
        base_dir, "data", f"chunk-{chunk:03d}", f"episode_{episode_index:06d}.parquet"
    )


def rewrite_parquet(src_path, dst_path, new_episode_index, global_index_offset):
    """Read a parquet file, remap episode_index and global index, write to dst."""
    table = pq.read_table(src_path)
    n = table.num_rows

    new_episode_col = pa.array([new_episode_index] * n, type=pa.int64())
    new_index_col = pa.array(
        list(range(global_index_offset, global_index_offset + n)), type=pa.int64()
    )

    col_idx_episode = table.schema.get_field_index("episode_index")
    col_idx_index = table.schema.get_field_index("index")

    table = table.set_column(col_idx_episode, "episode_index", new_episode_col)
    table = table.set_column(col_idx_index, "index", new_index_col)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    pq.write_table(table, dst_path)

    return n


def create_subset(src_dir, dst_base, suffix, shuffled_old_indices, episodes_meta, tasks_meta, info, stats_data):
    """Create one shuffled subset dataset."""
    n_episodes = len(shuffled_old_indices)
    dst_dir = os.path.join(dst_base, f"lerobot_all_repo_{suffix}")
    os.makedirs(os.path.join(dst_dir, "meta"), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Creating {suffix}: {n_episodes} episodes -> {dst_dir}")
    print(f"{'='*60}")

    new_episodes = []
    global_index = 0
    total_frames = 0
    task_set = set()
    t0 = time.time()

    for new_ep_idx, old_ep_idx in enumerate(shuffled_old_indices):
        old_meta = episodes_meta[old_ep_idx]
        src_path = episode_parquet_path(src_dir, old_ep_idx)
        dst_path = episode_parquet_path(dst_dir, new_ep_idx)

        n_frames = rewrite_parquet(src_path, dst_path, new_ep_idx, global_index)
        assert n_frames == old_meta["length"], (
            f"Mismatch: ep {old_ep_idx} expected {old_meta['length']} frames, got {n_frames}"
        )

        new_episodes.append({
            "episode_index": new_ep_idx,
            "tasks": old_meta["tasks"],
            "length": n_frames,
        })

        for t in old_meta["tasks"]:
            task_set.add(t)

        global_index += n_frames
        total_frames += n_frames

        if (new_ep_idx + 1) % 100 == 0 or (new_ep_idx + 1) == n_episodes:
            elapsed = time.time() - t0
            eps_per_sec = (new_ep_idx + 1) / elapsed
            eta = (n_episodes - new_ep_idx - 1) / eps_per_sec if eps_per_sec > 0 else 0
            print(f"  [{new_ep_idx+1}/{n_episodes}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    write_jsonl(os.path.join(dst_dir, "meta", "episodes.jsonl"), new_episodes)

    used_tasks = [t for t in tasks_meta if t["task"] in task_set]
    write_jsonl(os.path.join(dst_dir, "meta", "tasks.jsonl"), used_tasks)

    new_info = copy.deepcopy(info)
    new_info["total_episodes"] = n_episodes
    new_info["total_frames"] = total_frames
    new_info["total_tasks"] = len(used_tasks)
    new_info["total_chunks"] = math.ceil(n_episodes / CHUNK_SIZE)
    new_info["splits"] = {"train": f"0:{n_episodes}"}
    with open(os.path.join(dst_dir, "meta", "info.json"), "w") as f:
        json.dump(new_info, f, indent=4)

    with open(os.path.join(dst_dir, "meta", "stats.json"), "w") as f:
        json.dump(stats_data, f, indent=4)

    readme_src = os.path.join(src_dir, "README.md")
    if os.path.exists(readme_src):
        shutil.copy2(readme_src, os.path.join(dst_dir, "README.md"))

    print(f"  Done: {n_episodes} episodes, {total_frames} frames, {len(used_tasks)} tasks")
    return dst_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str,
                        default="/mnt/data/mqs/workspace/.cache/huggingface/lerobot/lerobot_all_repo")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--subsets", type=str, nargs="+",
                        default=None,
                        help="Which subsets to create, e.g. 'shuffled shuffled_5pct'. Default: all.")
    args = parser.parse_args()

    src_dir = args.src
    dst_base = os.path.dirname(src_dir)

    episodes_meta = read_jsonl(os.path.join(src_dir, "meta", "episodes.jsonl"))
    tasks_meta = read_jsonl(os.path.join(src_dir, "meta", "tasks.jsonl"))
    with open(os.path.join(src_dir, "meta", "info.json")) as f:
        info = json.load(f)
    with open(os.path.join(src_dir, "meta", "stats.json")) as f:
        stats_data = json.load(f)

    n_total = len(episodes_meta)
    print(f"Source: {src_dir}")
    print(f"Total episodes: {n_total}, Total tasks: {len(tasks_meta)}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total).tolist()

    print(f"\nShuffled order (first 20): {perm[:20]}")
    print(f"Shuffled order (last 10):  {perm[-10:]}")

    subsets_to_create = SUBSETS
    if args.subsets:
        subsets_to_create = [(s, r) for s, r in SUBSETS if s in args.subsets]

    for suffix, ratio in subsets_to_create:
        n_eps = max(1, round(n_total * ratio))
        selected = perm[:n_eps]
        create_subset(src_dir, dst_base, suffix, selected, episodes_meta, tasks_meta, info, stats_data)

    print("\nAll done!")
    print("\nSummary:")
    for suffix, ratio in subsets_to_create:
        n_eps = max(1, round(n_total * ratio))
        path = os.path.join(dst_base, f"lerobot_all_repo_{suffix}")
        print(f"  {suffix}: {n_eps} episodes -> {path}")


if __name__ == "__main__":
    main()
