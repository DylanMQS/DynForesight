#!/usr/bin/env python3
"""
使用 Wan2.2-TI2V-5B 的 VAE encoder 提取视频/图像特征。
仅加载 VAE，不加载 DiT/T5。运行前请将当前目录或 Wan2.2 仓库根目录加入 PYTHONPATH。

用法示例:
  # 从视频提取特征
  python extract_vae_features.py --ckpt_dir ./Wan2.2-TI2V-5B --input video.mp4 --output feats.pt

  # 从图像提取特征（视为 1 帧视频）
  python extract_vae_features.py --ckpt_dir ./Wan2.2-TI2V-5B --input frame.png --output feats.pt
"""
import argparse
import os
import sys
import time

import torch
import torchvision.io as tv_io
from PIL import Image

# 只加载 VAE 模块，不经过 wan 包根，避免拉取 speech2video 等依赖（如 librosa）
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
# 与 wan/configs/wan_ti2v_5B.py 一致
VAE_STRIDE = (4, 16, 16)
# 输入 H,W 建议为 32 的倍数（patch 2 + stride 16）
MIN_SIZE = 32


def _round_to_multiple(x, base):
    return max(base, (x // base) * base)


def load_image_as_tensor(path: str, device: torch.device, target_hw=None):
    """加载单张图像 -> (C, 1, H, W), 范围 [-1, 1]。"""
    img = Image.open(path).convert("RGB")
    x = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(1)  # (C, 1, H, W)
    if target_hw is not None:
        h, w = target_hw
        x = torch.nn.functional.interpolate(
            x, size=(h, w), mode="bilinear", align_corners=False
        )
    return x.to(device)


def load_video_as_tensor(path: str, device: torch.device, target_hw=None, max_frames=None):
    """加载视频 -> (C, T, H, W), 范围 [-1, 1]。"""
    # torchvision.io.read_video 返回 (T, H, W, C), uint8, 0-255
    v, _, _ = tv_io.read_video(path, pts_unit="sec")
    if v.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape {v.shape}")
    v = v.float() / 255.0
    v = (v - 0.5) / 0.5
    v = v.permute(3, 0, 1, 2)  # (T,H,W,C) -> (C, T, H, W)
    if max_frames is not None and v.shape[1] > max_frames:
        v = v[:, :max_frames]
    if target_hw is None:
        target_hw = (224, 224)
    h, w = target_hw
    if v.shape[2] != h or v.shape[3] != w:
        v = torch.nn.functional.interpolate(
            v, size=(h, w), mode="bilinear", align_corners=False
        )
    t = v.shape[1]
    aligned = ((t - 1) // 4) * 4 + 1
    v = v[:, :aligned]
    return v.to(device)


def get_video_shape(path: str):
    """返回视频的 (H, W)。read_video 格式为 (T, H, W, C)。"""
    v, _, _ = tv_io.read_video(path, pts_unit="sec")
    if v.ndim != 4 or v.numel() == 0:
        raise ValueError(f"Cannot get video shape: {v.shape}")
    return v.shape[1], v.shape[2]  # H, W


def main():
    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B VAE encoder 特征提取")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./Wan2.2-TI2V-5B",
        help="TI2V-5B 权重目录，内含 Wan2.2_VAE.pth",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入视频或图像路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出特征路径 (.pt)。默认在输入同目录下生成 xxx_vae_feats.pt",
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
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="视频最大帧数，超过则截断（仅视频）",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=224,
        help="空间最长边上限，保持比例缩放后再对齐到 32（0 表示不缩放，默认 224）",
    )
    args = parser.parse_args()

    vae_pth = os.path.join(args.ckpt_dir, VAE_CKPT_NAME)
    if not os.path.isfile(vae_pth):
        print(f"Error: VAE weight not found: {vae_pth}")
        print("Please download Wan2.2-TI2V-5B and ensure Wan2.2_VAE.pth is in --ckpt_dir.")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    # 加载 VAE
    vae = Wan2_2_VAE(vae_pth=vae_pth, device=device, dtype=dtype)
    vae.model.eval()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    # 根据扩展名判断图像/视频
    ext = os.path.splitext(input_path)[1].lower()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    is_image = ext in image_exts

    # 固定空间尺寸 224x224（H, W 均为 32 的倍数，满足 VAE 对齐要求）
    target_hw = (224, 224)

    # --- 加载输入 ---
    t_load_start = time.time()
    with torch.no_grad():
        if is_image:
            x = load_image_as_tensor(input_path, device, target_hw=target_hw)
        else:
            x = load_video_as_tensor(
                input_path, device, target_hw=target_hw, max_frames=args.max_frames
            )
    t_load_end = time.time()
    print(f"[Input]  shape: {tuple(x.shape)}  (C, T, H, W)")
    print(f"[Timer]  Load + preprocess: {t_load_end - t_load_start:.3f}s")

    # --- VAE encode (batch=16, 同一视频复制 16 份并行处理) ---
    batch_size = 1
    # x: (C, T, H, W) -> batch: (B, C, T, H, W)
    batch = x.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).contiguous()
    print(f"[Batch]  shape: {tuple(batch.shape)}  (B, C, T, H, W)")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_enc_start = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        vae.model.clear_cache()
        z_batch = vae.model.encode(batch, vae.scale)  # (B, z_dim, T', H', W')
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_enc_end = time.time()

    z = z_batch[0]  # 取第一个样本的结果
    print(f"[Output] single shape: {tuple(z.shape)}  (z_dim, T', H', W')")
    print(f"[Output] batch  shape: {tuple(z_batch.shape)}  (B, z_dim, T', H', W')")
    print(f"[Timer]  VAE encode (batch={batch_size}): {t_enc_end - t_enc_start:.3f}s")
    print(f"[Timer]  Per-sample: {(t_enc_end - t_enc_start) / batch_size:.3f}s")

    # --- 转移到 CPU + 保存 ---
    t_save_start = time.time()
    if z.device.type != "cpu":
        z = z.cpu()
    z = z.float()

    out_path = args.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(os.path.dirname(input_path), f"{base}_vae_feats.pt")
    torch.save({"latent": z, "shape": tuple(z.shape)}, out_path)
    t_save_end = time.time()
    print(f"[Timer]  Save to disk: {t_save_end - t_save_start:.3f}s")

    t_total = t_save_end - t_load_start
    print(f"[Timer]  Total (load + encode + save): {t_total:.3f}s")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()


# python extract_vae_features.py \
#   --ckpt_dir ./Wan2.2-TI2V-5B \
#   --input /mnt/data/mqs/workspace/VLA/episode_000000.mp4 \
#   --output /mnt/data/mqs/workspace/VLA/episode_000000_vae_feats.pt