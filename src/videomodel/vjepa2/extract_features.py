"""
VJEPA2 ViT-L video feature extraction using PyAV for video decoding.

Usage:
    conda run -n vjepa2 python extract_features.py \
        --video /mnt/data/mqs/workspace/VLA/episode_000000.mp4 \
        --checkpoint checkpoints/vitl.pt \
        --num_frames 16 \
        --output features.pt
"""

import argparse

import av
import numpy as np
import torch
import torch.nn.functional as F

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_large

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_video_av(video_path, num_frames):
    """Load video frames using PyAV, uniformly sample num_frames."""
    container = av.open(video_path)
    stream = container.streams.video[0]

    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")  # H, W, 3
        frames.append(img)
    container.close()

    total = len(frames)
    if total == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    sampled = np.stack([frames[i] for i in indices], axis=0)  # T, H, W, C
    print(f"Loaded {total} frames, sampled {num_frames} at indices: {indices.tolist()}")
    print(f"Frame shape: {sampled.shape}")
    return sampled


def build_eval_transform(crop_size=256):
    short_side_size = int(256.0 / 224 * crop_size)
    return video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(crop_size, crop_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames to sample (must be divisible by tubelet_size=2)")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save features (.pt file). If not set, only print shape.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for benchmarking")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    assert args.num_frames % 2 == 0, "num_frames must be divisible by tubelet_size=2"

    # --- 1. Load video ---
    video = load_video_av(args.video, args.num_frames)  # T, H, W, C

    # --- 2. Preprocess ---
    transform = build_eval_transform(args.crop_size)
    video_tensor = transform(video)                     # C, T, H, W
    video_tensor = video_tensor.unsqueeze(0)            # 1, C, T, H, W
    video_tensor = video_tensor.to(args.device)
    print(f"Input tensor shape: {video_tensor.shape}")

    # --- 3. Build encoder ---
    encoder = vit_large(
        img_size=(args.crop_size, args.crop_size),
        patch_size=16,
        num_frames=args.num_frames,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
    )

    # --- 4. Load weights ---
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    encoder_sd = state_dict["encoder"]
    encoder_sd = {k.replace("module.", "").replace("backbone.", ""): v
                  for k, v in encoder_sd.items()}
    msg = encoder.load_state_dict(encoder_sd, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  missing keys:    {msg.missing_keys}")
    print(f"  unexpected keys: {msg.unexpected_keys}")

    encoder = encoder.to(args.device).eval()

    # --- 5. Extract features + benchmark ---
    num_warmup = 3
    num_benchmark = 10
    batch_sizes = [1, args.batch_size] if args.batch_size > 1 else [1]

    for bs in batch_sizes:
        input_batch = video_tensor.expand(bs, -1, -1, -1, -1).contiguous()
        print(f"\n=== Encoder Timing (batch_size={bs}, {num_benchmark} runs, {num_warmup} warmup) ===")
        print(f"  Input shape: {input_batch.shape}")

        with torch.inference_mode():
            for _ in range(num_warmup):
                _ = encoder(input_batch)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            times = []
            for _ in range(num_benchmark):
                start_event.record()
                out = encoder(input_batch)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)
        print(f"  Average: {avg_ms:.2f} ms  ({avg_ms/bs:.2f} ms/video)")
        print(f"  Min:     {min_ms:.2f} ms")
        print(f"  Max:     {max_ms:.2f} ms")
        print(f"  Throughput: {bs * 1000.0 / avg_ms:.2f} videos/s")
        print(f"  VRAM used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Final run with bs=1 for saving
    with torch.inference_mode():
        features = encoder(video_tensor)

    print(f"\n=== Feature Extraction Results ===")
    print(f"Patch features shape: {features.shape}")

    T_tokens = args.num_frames // 2
    H_tokens = args.crop_size // 16
    W_tokens = args.crop_size // 16
    print(f"  = (batch=1, T={T_tokens} x H={H_tokens} x W={W_tokens} = {T_tokens*H_tokens*W_tokens}, dim=1024)")

    # Global average pooling
    global_feat = features.mean(dim=1)  # (1, 1024)
    print(f"Global feature (mean pooled): {global_feat.shape}")

    # Per-frame features (spatial avg)
    spatial_temporal = features.reshape(1, T_tokens, H_tokens, W_tokens, 1024)
    per_frame_feat = spatial_temporal.mean(dim=(2, 3))  # (1, T_tokens, 1024)
    print(f"Per-frame features: {per_frame_feat.shape}")

    # --- 6. Save ---
    if args.output:
        save_dict = {
            "patch_features": features.cpu(),           # (1, N, 1024)
            "global_feature": global_feat.cpu(),        # (1, 1024)
            "per_frame_features": per_frame_feat.cpu(), # (1, T_tokens, 1024)
            "video_path": args.video,
            "num_frames": args.num_frames,
            "crop_size": args.crop_size,
        }
        torch.save(save_dict, args.output)
        print(f"\nSaved features to: {args.output}")


if __name__ == "__main__":
    main()
