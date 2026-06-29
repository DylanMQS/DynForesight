"""
VJEPA2 feature compressor that maps encoder output to Wan VAE-compatible shape.

VJEPA2 raw:  (B, 8, 16, 16, 1024)   — 2,097,152 values
Wan VAE:     (B, 4, 14, 14, 48)     —    37,632 values
Compressed:  (B, 4, 14, 14, 48)     —    37,632 values  (55.7× compression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VJEPA2Compressor(nn.Module):
    """
    Compress VJEPA2 patch features to match Wan2.2 VAE latent shape.

    Input:  (B, T*H*W, encoder_dim)  e.g. (B, 2048, 1024)
    Output: (B, t_out*h_out*w_out, out_dim)  e.g. (B, 784, 48)

    Also provides structured output (B, t_out, h_out, w_out, out_dim).
    """

    def __init__(
        self,
        encoder_dim=1024,
        out_dim=48,
        t_in=8, h_in=16, w_in=16,
        t_out=4, h_out=14, w_out=14,
    ):
        super().__init__()
        self.t_in = t_in
        self.h_in = h_in
        self.w_in = w_in
        self.t_out = t_out
        self.h_out = h_out
        self.w_out = w_out
        self.out_dim = out_dim

        self.proj = nn.Linear(encoder_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, return_structured=False):
        """
        Args:
            x: (B, T*H*W, D) raw VJEPA2 encoder output
            return_structured: if True, return (B, t, h, w, out_dim)
                               if False, return (B, t*h*w, out_dim)
        """
        B, N, D = x.shape
        # Reshape to spatial-temporal grid
        x = x.reshape(B, self.t_in, self.h_in, self.w_in, D)
        x = x.permute(0, 4, 1, 2, 3)                     # (B, D, T, H, W)

        # 3D adaptive pooling — no learnable params, just interpolation
        x = F.adaptive_avg_pool3d(
            x, (self.t_out, self.h_out, self.w_out)
        )                                                   # (B, D, t, h, w)

        x = x.permute(0, 2, 3, 4, 1)                      # (B, t, h, w, D)

        # Linear projection: D → out_dim
        x = self.proj(x)                                   # (B, t, h, w, out_dim)

        if return_structured:
            return x                                        # (B, 4, 14, 14, 48)
        return x.reshape(B, -1, self.out_dim)              # (B, 784, 48)


def demo():
    """Quick demo comparing with Wan VAE output shape."""
    import argparse
    import av
    import numpy as np
    import src.datasets.utils.video.transforms as video_transforms
    import src.datasets.utils.video.volume_transforms as volume_transforms
    from src.models.vision_transformer import vit_large

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vitl.pt")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # --- Load video ---
    container = av.open(args.video)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    indices = np.linspace(0, len(frames) - 1, args.num_frames, dtype=int)
    video = np.stack([frames[i] for i in indices])

    crop_size = 256
    short_side = int(256.0 / 224 * crop_size)
    transform = video_transforms.Compose([
        video_transforms.Resize(short_side, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(crop_size, crop_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    video_tensor = transform(video).unsqueeze(0).to(args.device)

    # --- Encoder ---
    encoder = vit_large(
        img_size=(crop_size, crop_size), patch_size=16,
        num_frames=args.num_frames, tubelet_size=2,
        use_sdpa=True, use_SiLU=False, wide_SiLU=True,
        uniform_power=False, use_rope=True,
    )
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["encoder"]
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    encoder.load_state_dict(sd, strict=False)
    encoder = encoder.to(args.device).eval()

    # --- Compressor ---
    T = args.num_frames // 2
    compressor = VJEPA2Compressor(
        encoder_dim=1024, out_dim=48,
        t_in=T, h_in=16, w_in=16,
        t_out=4, h_out=14, w_out=14,
    ).to(args.device)

    # --- Forward ---
    with torch.inference_mode():
        raw = encoder(video_tensor)
        compressed_flat = compressor(raw, return_structured=False)
        compressed_struct = compressor(raw, return_structured=True)

    print("=" * 60)
    print("VJEPA2 Feature Compression → Wan VAE Compatible Shape")
    print("=" * 60)
    print(f"")
    print(f"  VJEPA2 encoder output:  {raw.shape}")
    print(f"    reshaped:             (1, {T}, 16, 16, 1024)")
    print(f"    total values:         {raw.numel():,}")
    print(f"")
    print(f"  Compressed (flat):      {compressed_flat.shape}")
    print(f"  Compressed (structured):{compressed_struct.shape}")
    print(f"    total values:         {compressed_flat.numel():,}")
    print(f"")
    print(f"  Wan VAE output:         (1, 48, 4, 14, 14)")
    print(f"    = structured:         (1, 4, 14, 14, 48)")
    print(f"    = flat:               (1, 784, 48)")
    print(f"    total values:         37,632")
    print(f"")
    print(f"  Compression ratio:      {raw.numel() / compressed_flat.numel():.1f}×")
    print(f"  Shape match:            ✓" if compressed_struct.shape == (1, 4, 14, 14, 48) else "  Shape match: ✗")


if __name__ == "__main__":
    demo()
