#!/usr/bin/env python3
"""
使用 Wan2.2-TI2V-5B 的 VAE decoder 从 latent 特征恢复视频。
仅加载 VAE，不加载 DiT/T5。

用法示例:
  python decode_vae_features.py \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --input episode_000000_vae_feats.pt \
    --output decoded_video.mp4 \
    --fps 24
"""
import argparse
import os
import sys
import time

import torch
import torchvision.io as tv_io


def _load_vae_class():
    _root = os.path.dirname(os.path.abspath(__file__))
    vae2_2_path = os.path.join(_root, "wan", "modules", "vae2_2.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("vae2_2_standalone", vae2_2_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Wan2_2_VAE


Wan2_2_VAE = _load_vae_class()

VAE_CKPT_NAME = "Wan2.2_VAE.pth"


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 VAE decoder: latent -> video")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./Wan2.2-TI2V-5B",
        help="TI2V-5B 权重目录，内含 Wan2.2_VAE.pth",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="VAE latent 特征文件 (.pt)，由 extract_vae_features.py 生成",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出视频路径 (.mp4)。默认在输入同目录下生成 xxx_decoded.mp4",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="输出视频帧率 (默认 24)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备: cuda 或 cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("float32", "float16", "bfloat16"),
        help="VAE 计算 dtype",
    )
    args = parser.parse_args()

    vae_pth = os.path.join(args.ckpt_dir, VAE_CKPT_NAME)
    if not os.path.isfile(vae_pth):
        print(f"Error: VAE weight not found: {vae_pth}")
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    # --- 加载 latent ---
    t0 = time.time()
    data = torch.load(args.input, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "latent" in data:
        z = data["latent"]
    elif isinstance(data, torch.Tensor):
        z = data
    else:
        print(f"Error: 无法识别的 .pt 格式，keys={list(data.keys()) if isinstance(data, dict) else type(data)}")
        sys.exit(1)
    print(f"[Latent] shape: {tuple(z.shape)}  (z_dim, T', H', W')")
    print(f"[Timer]  Load latent: {time.time() - t0:.3f}s")

    # --- 加载 VAE ---
    t1 = time.time()
    vae = Wan2_2_VAE(vae_pth=vae_pth, device=device, dtype=dtype)
    vae.model.eval()
    print(f"[Timer]  Load VAE model: {time.time() - t1:.3f}s")

    # --- Decode ---
    z = z.to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        vae.model.clear_cache()
        # WanVAE_.decode: 输入 (B, z_dim, T', H', W'), scale; 输出 (B, C, T, H, W)
        video = vae.model.decode(z.unsqueeze(0), vae.scale)  # (1, C, T, H, W)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.time()
    print(f"[Output] decoded shape: {tuple(video.shape)}  (B, C, T, H, W)")
    print(f"[Timer]  VAE decode: {t3 - t2:.3f}s")

    # --- 后处理: [-1, 1] -> [0, 255] uint8, (T, H, W, C) ---
    video = video[0].float().clamp(-1, 1)       # (C, T, H, W)
    video = (video + 1.0) / 2.0 * 255.0         # [0, 255]
    video = video.permute(1, 2, 3, 0)            # (T, H, W, C)
    video = video.to("cpu", dtype=torch.uint8)
    print(f"[Video]  frames={video.shape[0]}, H={video.shape[1]}, W={video.shape[2]}")

    # --- 保存 ---
    out_path = args.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(os.path.dirname(args.input) or ".", f"{base}_decoded.mp4")

    t4 = time.time()
    tv_io.write_video(out_path, video, fps=args.fps)
    print(f"[Timer]  Save video: {time.time() - t4:.3f}s")
    print(f"Saved decoded video to {out_path}")


if __name__ == "__main__":
    main()


# python decode_vae_features.py \
#   --ckpt_dir ./Wan2.2-TI2V-5B \
#   --input /mnt/workspace/mqs/workspace/VLA/episode_000000_vae_feats.pt \
#   --output /mnt/workspace/mqs/workspace/VLA/episode_000000_decoded.mp4
