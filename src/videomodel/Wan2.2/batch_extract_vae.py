#!/usr/bin/env python3
"""
多卡并行批量提取 Wan2.2-TI2V-5B VAE 特征。

从 steps_data_index.pkl 读取所有 (trajectory_id, base_index)，
对每条记录从对应视频中取 base_index 及其后 16 帧（共 17 帧，满足 VAE 的 4k+1 要求），
送入 VAE encoder，输出 latent 保存到目标目录。

用法（直接 python，无需 torchrun）:
  python batch_extract_vae.py --num_gpus 8 --batch_size 16
  python batch_extract_vae.py --num_gpus 1 --batch_size 8   # 单卡测试
"""
import argparse
import gc
import os
import pickle
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_STOP_FLAG = False


def _sigterm_handler(signum, frame):
    global _STOP_FLAG
    _STOP_FLAG = True
    print(f"\n[INFO] Received signal {signum}, will stop after current batch...")


signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT, _sigterm_handler)


def _load_vae_class():
    _root = os.path.dirname(os.path.abspath(__file__))
    vae2_2_path = os.path.join(_root, "wan", "modules", "vae2_2.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("vae2_2_standalone", vae2_2_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Wan2_2_VAE


Wan2_2_VAE = _load_vae_class()

CHUNK_SIZE = 1000
VIDEO_PATH_PATTERN = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
TARGET_HW = (224, 224)


def get_video_path(video_root: str, trajectory_id: int, video_key: str) -> str:
    chunk_index = trajectory_id // CHUNK_SIZE
    return os.path.join(
        video_root,
        VIDEO_PATH_PATTERN.format(
            episode_chunk=chunk_index,
            episode_index=trajectory_id,
            video_key=video_key,
        ),
    )


def decode_frames(video_path: str, start_frame: int, num_frames: int) -> torch.Tensor:
    """用 pyav 解码指定帧 -> (C, T, H, W), [-1, 1]。
    超出视频末尾的帧用 0 填充。
    对 start_frame > 0 的情况先 seek 到附近关键帧，避免逐帧扫描。"""
    import av

    container = None
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        if start_frame > 30 and stream.average_rate:
            fps = float(stream.average_rate)
            time_base = float(stream.time_base) if stream.time_base else 1.0 / fps
            target_ts = int(max(0, start_frame - 5) / fps / time_base)
            container.seek(target_ts, stream=stream, backward=True)

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if hasattr(frame, 'pts') and stream.average_rate:
                frame_idx = int(frame.pts * float(stream.time_base) * float(stream.average_rate))
            else:
                frame_idx = i
            if frame_idx < start_frame:
                continue
            if frame_idx >= start_frame + num_frames:
                break
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
            if len(frames) >= num_frames:
                break
    except Exception as e:
        raise RuntimeError(f"pyav decode error for {video_path} at frame {start_frame}: {e}")
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from {video_path} at frame {start_frame}")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    h, w = TARGET_HW
    t = torch.from_numpy(np.stack(frames)).float() / 255.0  # (T, H, W, C)
    t = (t - 0.5) / 0.5
    t = t.permute(3, 0, 1, 2)  # (C, T, H, W)

    if t.shape[2] != h or t.shape[3] != w:
        t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)

    return t


class VideoSegmentDataset(Dataset):
    """每个样本: 从 episode 视频中取 [base_index, base_index + num_frames) 共 num_frames 帧。"""

    def __init__(self, steps, video_root, video_key, num_frames):
        self.steps = steps
        self.video_root = video_root
        self.video_key = video_key
        self.num_frames = num_frames

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        trajectory_id, base_index = self.steps[idx]
        video_path = get_video_path(self.video_root, trajectory_id, self.video_key)
        try:
            frames = decode_frames(video_path, base_index, self.num_frames)
        except Exception as e:
            print(f"[WARN] Failed to decode {video_path} frame {base_index}: {e}")
            frames = torch.zeros(3, self.num_frames, *TARGET_HW)
        return frames, trajectory_id, base_index


def get_output_path(output_dir: str, video_key: str, trajectory_id: int, base_index: int) -> str:
    chunk_index = trajectory_id // CHUNK_SIZE
    sub_dir = os.path.join(output_dir, f"chunk-{chunk_index:03d}")
    return os.path.join(sub_dir, video_key, f"{trajectory_id}_{base_index}.pt")


def _save_tensor(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)


def main():
    parser = argparse.ArgumentParser(description="8卡并行 VAE 特征提取")
    parser.add_argument("--ckpt_dir", type=str, default="./Wan2.2-TI2V-5B")
    parser.add_argument(
        "--steps_pkl",
        type=str,
        default="/mnt/data/mqs/workspace/VLA/starVLA/playground/Datasets/droid_lerobot/meta/steps_data_index.pkl",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="/mnt/data/mqs/workspace/VLA/starVLA/playground/Datasets/droid_lerobot",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/data/mqs/workspace/VLA/starVLA/playground/Datasets/droid_lerobot/vae_features",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_frames", type=int, default=17,
                        help="从 base_index 起读取的总帧数（含 base_index 自身）。"
                             "必须为 4k+1（如 5,9,13,17,21）。默认 17 = base_index + 后续 16 帧")
    parser.add_argument(
        "--video_key",
        type=str,
        # default="observation.images.exterior_image_1_left",
        default="observation.images.wrist_image_left",
        help="视频的 key（对应 videos/ 下的子目录名）",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker 数 (每个 GPU)")
    parser.add_argument("--num_gpus", type=int, default=8, help="使用的 GPU 数量")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已存在的特征文件（默认开启，实现断点续传）")
    parser.add_argument("--no_skip_existing", action="store_true", help="不跳过已存在的特征文件")
    parser.add_argument("--save_dtype", type=str, default="float32",
                        choices=("float32", "bfloat16"), help="保存特征的 dtype (bfloat16 可减半文件大小)")
    parser.add_argument("--save_threads", type=int, default=4, help="异步保存线程数")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor")
    args = parser.parse_args()

    # --- 加载 steps（主进程加载一次，传给子进程） ---
    with open(args.steps_pkl, "rb") as f:
        cached = pickle.load(f)
    if isinstance(cached, dict) and "steps" in cached:
        all_steps = cached["steps"]
    elif isinstance(cached, list):
        all_steps = cached
    else:
        raise ValueError(f"Unexpected pickle format: {type(cached)}")

    if args.no_skip_existing:
        args.skip_existing = False

    if (args.num_frames - 1) % 4 != 0:
        corrected = ((args.num_frames - 1 + 3) // 4) * 4 + 1
        print(f"[WARNING] num_frames={args.num_frames} is not 4k+1. "
              f"VAE encoder processes frames as 1+4+4+..., non-4k+1 values cause silent frame loss. "
              f"Auto-correcting to {corrected}.")
        args.num_frames = corrected

    print(f"Total steps: {len(all_steps)}")

    # 跳过已存在（默认开启，实现断点续传）
    if args.skip_existing:
        filtered = []
        for traj_id, base_idx in all_steps:
            out_path = get_output_path(args.output_dir, args.video_key, traj_id, base_idx)
            if not os.path.exists(out_path):
                filtered.append((traj_id, base_idx))
        print(f"After skipping existing: {len(filtered)} steps (skipped {len(all_steps) - len(filtered)})")
        all_steps = filtered

    # 按 trajectory_id 排序，提升同一视频文件的 OS page cache 命中率
    all_steps.sort(key=lambda x: (x[0], x[1]))

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    chunk_ids = set(t[0] // CHUNK_SIZE for t in all_steps)
    for cid in chunk_ids:
        os.makedirs(os.path.join(args.output_dir, f"chunk-{cid:03d}"), exist_ok=True)

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    total_workers = num_gpus * args.num_workers
    print(f"Using {num_gpus} GPU(s), batch_size per GPU: {args.batch_size}")
    print(f"Effective batch_size: {num_gpus * args.batch_size}")
    print(f"Total DataLoader workers: {total_workers} ({args.num_workers} per GPU)")

    if num_gpus == 1:
        worker(0, 1, all_steps, args)
    else:
        try:
            mp.set_start_method("forkserver", force=True)
        except RuntimeError:
            pass
        mp.spawn(worker, nprocs=num_gpus, args=(num_gpus, all_steps, args))


def worker(rank, world_size, all_steps, args):
    global _STOP_FLAG

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dtype = getattr(torch, args.dtype)
    save_dtype = getattr(torch, args.save_dtype)

    per_rank = len(all_steps) // world_size
    start = rank * per_rank
    end = start + per_rank if rank < world_size - 1 else len(all_steps)
    my_steps = all_steps[start:end]

    vae_pth = os.path.join(args.ckpt_dir, "Wan2.2_VAE.pth")
    vae = Wan2_2_VAE(vae_pth=vae_pth, device=device, dtype=dtype)
    vae.model.eval()

    if rank == 0:
        print(f"VAE loaded on {world_size} GPU(s), per-GPU steps: ~{per_rank}")
        print(f"Save dtype: {args.save_dtype}, save threads: {args.save_threads}")
        print(f"DataLoader: num_workers={args.num_workers}, prefetch_factor={args.prefetch_factor}")

    dataset = VideoSegmentDataset(my_steps, args.video_root, args.video_key, args.num_frames)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        multiprocessing_context="forkserver" if args.num_workers > 0 else None,
    )

    save_pool = ThreadPoolExecutor(max_workers=args.save_threads)
    save_futures = []

    t_start = time.time()
    processed = 0
    errors = 0
    log_interval = 5000

    pbar = tqdm(
        total=len(my_steps),
        desc=f"[GPU{rank}]",
        disable=(rank != 0),
        unit="sample",
        dynamic_ncols=True,
    )

    try:
        for frames_batch, traj_ids, base_indices in loader:
            if _STOP_FLAG:
                if rank == 0:
                    print(f"\n[GPU{rank}] Graceful stop requested, finishing current saves...")
                break

            B = frames_batch.shape[0]

            try:
                frames_batch = frames_batch.to(device, non_blocking=True)

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                    vae.model.clear_cache()
                    z_batch = vae.model.encode(frames_batch, vae.scale)

                z_batch = z_batch.to(save_dtype).cpu()
            except Exception as e:
                errors += B
                if rank == 0:
                    print(f"\n[GPU{rank}][ERROR] VAE encode failed: {e}")
                pbar.update(B)
                processed += B
                del frames_batch
                torch.cuda.empty_cache()
                continue

            for i in range(B):
                traj_id = traj_ids[i].item()
                base_idx = base_indices[i].item()
                out_path = get_output_path(args.output_dir, args.video_key, traj_id, base_idx)
                tensor = z_batch[i].clone()
                save_futures.append(save_pool.submit(_save_tensor, tensor, out_path))

            if len(save_futures) > 500:
                done = [f for f in save_futures if f.done()]
                for f in done:
                    try:
                        f.result()
                    except Exception as e:
                        errors += 1
                        if rank == 0:
                            print(f"\n[GPU{rank}][ERROR] Save failed: {e}")
                save_futures = [f for f in save_futures if not f.done()]

            processed += B
            pbar.update(B)

            if rank == 0 and processed % log_interval < B:
                elapsed = time.time() - t_start
                throughput = processed / elapsed if elapsed > 0 else 0
                print(f"\n[GPU{rank}][CHECKPOINT] {processed}/{len(my_steps)} "
                      f"({100*processed/len(my_steps):.1f}%), "
                      f"{throughput:.1f} sample/s, errors: {errors}, "
                      f"elapsed: {elapsed:.0f}s")

            del frames_batch, z_batch
    except Exception as e:
        print(f"\n[GPU{rank}][FATAL] Worker loop error: {e}")
        traceback.print_exc()
    finally:
        for f in save_futures:
            try:
                f.result(timeout=60)
            except Exception as e:
                if rank == 0:
                    print(f"[GPU{rank}][ERROR] Save failed during cleanup: {e}")
        save_pool.shutdown(wait=True)

        del loader
        del dataset
        gc.collect()

        pbar.close()
        elapsed = time.time() - t_start

        if rank == 0:
            stopped_reason = "STOPPED (signal)" if _STOP_FLAG else "Done"
            print(f"\n{stopped_reason}! GPU{rank} processed {processed}/{len(my_steps)} steps "
                  f"in {elapsed:.1f}s, errors: {errors}")
            if not _STOP_FLAG:
                total_all = len(all_steps)
                print(f"Throughput: ~{total_all / elapsed:.1f} samples/s (all GPUs)")
            print(f"Output: {args.output_dir}")
            print(f"[TIP] Re-run with --skip_existing (default) to resume from where you left off.")


if __name__ == "__main__":
    main()
