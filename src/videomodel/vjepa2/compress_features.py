"""
VJEPA2 feature compression strategies.

Compresses raw patch features (B, T*H*W, 1024) into compact representations
suitable for alignment with VLM learnable queries.

Usage:
    conda run -n vjepa2 python compress_features.py \
        --video /mnt/data/mqs/workspace/VLA/episode_000000.mp4 \
        --checkpoint checkpoints/vitl.pt \
        --num_frames 16
"""

import argparse
import time

import av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_large

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_video_av(video_path, num_frames):
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return np.stack([frames[i] for i in indices], axis=0)


def build_eval_transform(crop_size=256):
    short_side_size = int(256.0 / 224 * crop_size)
    return video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(crop_size, crop_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


# =====================================================================
#  Compression Strategies
# =====================================================================

def compress_spatial_pool(features, T, H, W, spatial_size=4):
    """
    Strategy A: 空间池化，保留全部时间步
    (B, T*H*W, D) → (B, T*s*s, D)

    Example: T=8, H=W=16, spatial_size=4
      (1, 2048, 1024) → (1, 8*4*4, 1024) = (1, 128, 1024)
    """
    B, _, D = features.shape
    x = features.reshape(B, T, H, W, D)
    x = x.permute(0, 1, 4, 2, 3)                     # (B, T, D, H, W)
    x = x.reshape(B * T, D, H, W)
    x = F.adaptive_avg_pool2d(x, (spatial_size, spatial_size))
    x = x.reshape(B, T, D, spatial_size, spatial_size)
    x = x.permute(0, 1, 3, 4, 2)                     # (B, T, s, s, D)
    return x.reshape(B, T * spatial_size * spatial_size, D)


def compress_spatiotemporal_pool(features, T, H, W, t_size=4, s_size=4):
    """
    Strategy B: 时空联合池化
    (B, T*H*W, D) → (B, t*s*s, D)

    Example: T=8, H=W=16, t_size=4, s_size=4
      (1, 2048, 1024) → (1, 4*4*4, 1024) = (1, 64, 1024)
    """
    B, _, D = features.shape
    x = features.reshape(B, T, H, W, D).permute(0, 4, 1, 2, 3)  # (B, D, T, H, W)
    x = F.adaptive_avg_pool3d(x, (t_size, s_size, s_size))       # (B, D, t, s, s)
    x = x.permute(0, 2, 3, 4, 1)                                  # (B, t, s, s, D)
    return x.reshape(B, t_size * s_size * s_size, D)


def compress_spatial_pool_then_proj(features, T, H, W, spatial_size=4, out_dim=256):
    """
    Strategy C: 空间池化 + 维度投影
    (B, T*H*W, 1024) → (B, T*s*s, out_dim)

    Example: T=8, spatial_size=4, out_dim=256
      (1, 2048, 1024) → (1, 128, 256)
    """
    B, _, D = features.shape
    x = compress_spatial_pool(features, T, H, W, spatial_size)     # (B, N, 1024)
    proj = nn.Linear(D, out_dim).to(features.device).to(features.dtype)
    return proj(x)


def compress_spatiotemporal_pool_then_proj(features, T, H, W, t_size=4, s_size=4, out_dim=48):
    """
    Strategy D: 时空池化 + 维度投影 (最接近 Wan VAE 格式)
    (B, T*H*W, 1024) → (B, t*s*s, out_dim)

    Example: t_size=4, s_size=4, out_dim=48
      (1, 2048, 1024) → (1, 64, 48) — 对标 Wan VAE 的 (4,14,14,48)
    """
    B, _, D = features.shape
    x = compress_spatiotemporal_pool(features, T, H, W, t_size, s_size)
    proj = nn.Linear(D, out_dim).to(features.device).to(features.dtype)
    return proj(x)


def compress_conv3d(features, T, H, W, t_size=4, s_size=4, out_dim=256):
    """
    Strategy E: 3D卷积压缩 (可学习的下采样，比池化保留更多信息)
    (B, T*H*W, 1024) → (B, t*s*s, out_dim)
    """
    B, _, D = features.shape
    x = features.reshape(B, T, H, W, D).permute(0, 4, 1, 2, 3)  # (B, D, T, H, W)

    t_stride = T // t_size
    s_stride = H // s_size
    conv = nn.Conv3d(D, out_dim, kernel_size=(t_stride, s_stride, s_stride),
                     stride=(t_stride, s_stride, s_stride)).to(features.device).to(features.dtype)
    x = conv(x)                                                     # (B, out_dim, t, s, s)
    x = x.permute(0, 2, 3, 4, 1)                                   # (B, t, s, s, out_dim)
    return x.reshape(B, t_size * s_size * s_size, out_dim)


def compress_perceiver(features, num_queries=64, out_dim=256, num_heads=8):
    """
    Strategy F: Perceiver-style cross-attention (最灵活)
    用 N 个 learnable queries 通过 cross-attention 从 2048 tokens 中提取信息
    (B, 2048, 1024) → (B, num_queries, out_dim)
    """
    B, N, D = features.shape
    device, dtype = features.device, features.dtype

    queries = nn.Parameter(torch.randn(1, num_queries, D) * 0.02).to(device).to(dtype)
    cross_attn = nn.MultiheadAttention(D, num_heads, batch_first=True).to(device).to(dtype)
    proj = nn.Linear(D, out_dim).to(device).to(dtype)

    q = queries.expand(B, -1, -1)
    out, _ = cross_attn(q, features, features)  # (B, num_queries, D)
    return proj(out)                             # (B, num_queries, out_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load & preprocess
    video = load_video_av(args.video, args.num_frames)
    transform = build_eval_transform(args.crop_size)
    video_tensor = transform(video).unsqueeze(0).to(args.device)

    # Build & load encoder
    encoder = vit_large(
        img_size=(args.crop_size, args.crop_size), patch_size=16,
        num_frames=args.num_frames, tubelet_size=2,
        use_sdpa=True, use_SiLU=False, wide_SiLU=True,
        uniform_power=False, use_rope=True,
    )
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["encoder"]
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    encoder.load_state_dict(sd, strict=False)
    encoder = encoder.to(args.device).eval()

    # Extract raw features
    with torch.inference_mode():
        features = encoder(video_tensor)

    T = args.num_frames // 2
    H = W = args.crop_size // 16

    print(f"Raw features: {features.shape}  →  {features.numel():,} values\n")
    print(f"{'Strategy':<45} {'Shape':<25} {'Tokens':<8} {'Dim':<6} {'Total Values':<15} {'Compression'}")
    print("=" * 130)

    strategies = [
        ("A: spatial pool 4×4",
         lambda f: compress_spatial_pool(f, T, H, W, spatial_size=4)),
        ("A: spatial pool 2×2",
         lambda f: compress_spatial_pool(f, T, H, W, spatial_size=2)),
        ("B: spatiotemporal pool t=4,s=4",
         lambda f: compress_spatiotemporal_pool(f, T, H, W, t_size=4, s_size=4)),
        ("B: spatiotemporal pool t=4,s=2",
         lambda f: compress_spatiotemporal_pool(f, T, H, W, t_size=4, s_size=2)),
        ("C: spatial pool 4×4 + proj→256",
         lambda f: compress_spatial_pool_then_proj(f, T, H, W, spatial_size=4, out_dim=256)),
        ("D: st-pool t=4,s=4 + proj→48 (≈Wan VAE)",
         lambda f: compress_spatiotemporal_pool_then_proj(f, T, H, W, t_size=4, s_size=4, out_dim=48)),
        ("E: Conv3D → t=4,s=4, dim=256",
         lambda f: compress_conv3d(f, T, H, W, t_size=4, s_size=4, out_dim=256)),
        ("F: Perceiver 64 queries, dim=256",
         lambda f: compress_perceiver(f, num_queries=64, out_dim=256)),
        ("F: Perceiver 32 queries, dim=256",
         lambda f: compress_perceiver(f, num_queries=32, out_dim=256)),
    ]

    raw_total = features.numel()
    for name, fn in strategies:
        with torch.inference_mode():
            out = fn(features)
        total = out.numel()
        ratio = raw_total / total
        print(f"{name:<45} {str(tuple(out.shape)):<25} {out.shape[1]:<8} {out.shape[2]:<6} {total:<15,} {ratio:.1f}×")


if __name__ == "__main__":
    main()
