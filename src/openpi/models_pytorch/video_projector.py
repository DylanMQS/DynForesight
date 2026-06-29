"""Projector for aligning VLA vision hidden states with video VAE latent features.

All multi-frame projectors (``multi_frame = True``) share two design principles:

1. **Blind prediction** — the VLA projection does NOT see the real VAE features
   during prediction; only temporal position cues are provided.
2. **Per-frame supervision** — losses are computed against every individual frame,
   giving T' × denser gradients than single-target alignment.

Projectors
----------
Baseline / backward-compatible:

- ``VideoAlignProjector``        — ``"mean"``  (default)
- ``MultiFrameCosineProjector``  — ``"max_cosine"`` / ``"softmax_cosine"``

Temporal reconstruction (blind per-frame prediction):

- ``TemporalReconstructAlignProjector`` — ``"temporal_reconstruct"``
  MLP decoder: ``VLA_proj + pos(t) → predicted[t]``.
- ``TemporalFlowAlignProjector``        — ``"temporal_flow"``
  Predicts inter-frame *dynamics* Δ_t plus static anchor.
- ``AutoregressiveAlignProjector``      — ``"autoregressive"``
  GRU unrolls from VLA_proj to predict frames sequentially.
- ``VariationalTemporalAlignProjector`` — ``"variational"``
  VAE bottleneck: VLA → μ, σ → z ~ N(μ, σ) → decode per frame; KL regulariser.
- ``ContrastiveTemporalAlignProjector`` — ``"contrastive"``
  InfoNCE: VLA patch must discriminate its own temporal features from those
  of other spatial locations.

Cross-attention (has access to VAE KV — stronger architecture, weaker constraint):

- ``CrossAttentionReconstructProjector`` — ``"cross_attention"``
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _sinusoidal_temporal_embed(T: int, dim: int, device: torch.device) -> torch.Tensor:
    """Fixed sinusoidal positional embedding for temporal positions 0..T-1.

    Returns [T, dim].  Works for arbitrary T without learnable parameters.
    """
    pos = torch.arange(T, device=device, dtype=torch.float32)
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -(math.log(10000.0) / half))
    phase = pos[:, None] * freq[None, :]
    return torch.cat([phase.sin(), phase.cos()], dim=-1)


def _sinusoidal_delta_embed(deltas: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for **signed** temporal deltas.

    Unlike ``_sinusoidal_temporal_embed`` which uses indices 0..T-1, this
    operates on actual delta values (e.g. ``[-1.5, 0.0, 2.5]``).  The
    sign naturally encodes direction (past < 0, current ≈ 0, future > 0)
    and the magnitude encodes temporal distance.
    """
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=deltas.device, dtype=torch.float32) * -(math.log(10000.0) / half))
    phase = deltas.float()[:, None] * freq[None, :]
    return torch.cat([phase.sin(), phase.cos()], dim=-1)


def _make_sincos_pos_embed_1d(dim: int, positions: torch.Tensor, omega_0: float = 100.0) -> torch.Tensor:
    """Sinusoidal embedding for 1-D positions (matches VGGT / SF paper).

    Returns [len(positions), dim].
    """
    assert dim % 2 == 0
    half = dim // 2
    omega = torch.arange(half, device=positions.device, dtype=torch.float64) / half
    omega = 1.0 / (omega_0 ** omega)  # decreasing frequencies
    phase = positions.reshape(-1).to(torch.float64)[:, None] * omega[None, :]
    return torch.cat([phase.sin(), phase.cos()], dim=-1).float()


def _create_uv_grid(W: int, H: int, aspect_ratio: float = 1.0,
                     dtype: torch.dtype = torch.float32,
                     device: torch.device | str = "cpu") -> torch.Tensor:
    """Create a [H, W, 2] UV coordinate grid (matches SF paper's normalisation).

    Spans are normalised by the diagonal so that UV magnitudes stay bounded
    regardless of aspect ratio.
    """
    diag = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag
    span_y = 1.0 / diag
    left_x = -span_x * (W - 1) / W
    right_x = span_x * (W - 1) / W
    top_y = -span_y * (H - 1) / H
    bottom_y = span_y * (H - 1) / H
    xs = torch.linspace(left_x, right_x, W, dtype=dtype, device=device)
    ys = torch.linspace(top_y, bottom_y, H, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack([uu, vv], dim=-1)  # [H, W, 2]


def _position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int,
                              omega_0: float = 100.0) -> torch.Tensor:
    """Convert 2-D position grid [H, W, 2] to sinusoidal embeddings [H, W, embed_dim]."""
    H, W, _ = pos_grid.shape
    pos_flat = pos_grid.reshape(-1, 2)
    emb_x = _make_sincos_pos_embed_1d(embed_dim // 2, pos_flat[:, 0], omega_0)
    emb_y = _make_sincos_pos_embed_1d(embed_dim // 2, pos_flat[:, 1], omega_0)
    return torch.cat([emb_x, emb_y], dim=-1).view(H, W, embed_dim)


def apply_spatial_pe(x: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    """Add 2-D sinusoidal UV positional embedding to spatial feature maps.

    Args:
        x: [B, C, H, W] feature maps (e.g. VAE latent after spatial reshape).
        ratio: scaling factor for the PE (paper uses 0.1).

    Returns:
        x + PE, same shape and dtype as input.
    """
    orig_dtype = x.dtype
    _, C, H, W = x.shape
    pos_embed = _create_uv_grid(W, H, aspect_ratio=1.0, dtype=torch.float32, device=x.device)
    pos_embed = _position_grid_to_embed(pos_embed, C)          # [H, W, C]
    pos_embed = (pos_embed * ratio).permute(2, 0, 1)[None]     # [1, C, H, W]
    return (x.float() + pos_embed).to(orig_dtype)


def _compute_vae_representative_deltas(video_delta_frames: list[int] | tuple[int, ...]) -> list[float]:
    """Compute the representative temporal delta for each VAE temporal feature.

    The video VAE compresses with a **4n+1** pattern:

    - The 1st input frame is compressed independently → 1 VAE feature.
    - Every subsequent group of 4 frames is compressed into 1 VAE feature.

    So ``len(video_delta_frames) = 4n+1`` frames produce ``n+1`` VAE temporal
    features (T').  The representative delta for each feature is the mean of
    the frame deltas it covers.

    Examples::

        [0,1,2,3,4,5,6,7,8]       → [0.0, 2.5, 6.5]       (T'=3)
        [-4,-3,-2,-1,0,1,2,3,4]   → [-4.0, -1.5, 2.5]     (T'=3)
        [-8,...,-1,0]              → [-8.0, -5.5, -1.5]     (T'=3)
        [0,1,...,16]               → [0.0, 2.5, 6.5, 10.5, 14.5]  (T'=5)
    """
    deltas = list(video_delta_frames)
    if not deltas:
        return [0.0]
    result = [float(deltas[0])]
    remaining = deltas[1:]
    for i in range(0, len(remaining), 4):
        group = remaining[i : i + 4]
        result.append(sum(group) / len(group))
    return result


def _masked_cosine_loss(cos_sim: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Compute ``(1 - cos_sim)`` averaged over valid tokens and batch.

    Args:
        cos_sim: [B, N]
        mask:    [B, N] bool or None
    """
    if mask is not None:
        cos_sim = cos_sim * mask.float()
        loss = (1.0 - cos_sim).sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1.0)
    else:
        loss = (1.0 - cos_sim).mean(dim=-1)
    return loss.mean()


def _init_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _decompose_cameras(n: int) -> tuple[int, int, int]:
    """Decompose *n* tokens into (G, H, W) with H == W and G*H*W == n.

    Finds the largest perfect-square factor of *n* to use as the per-camera
    spatial grid.  Returns ``(n_cameras, side, side)``.

    Examples: 256 → (1, 16, 16);  512 → (2, 16, 16);  768 → (3, 16, 16).
    """
    side = int(n ** 0.5)
    if side * side == n:
        return 1, side, side
    for s in range(side, 0, -1):
        if n % (s * s) == 0:
            return n // (s * s), s, s
    return n, 1, 1


def _make_vla_proj_layers(vla_dim: int, vae_dim: int, use_vla_norm: bool):
    """Return (fc1, act, fc2, vla_norm) — the standard VLA→VAE projection."""
    fc1 = nn.Linear(vla_dim, vae_dim * 4)
    act = nn.GELU()
    fc2 = nn.Linear(vae_dim * 4, vae_dim)
    vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
    return fc1, act, fc2, vla_norm


def _project_vla(hidden, fc1, act, fc2, vla_norm):
    if vla_norm is not None:
        hidden = vla_norm(hidden)
    return fc2(act(fc1(hidden)))


def _make_temporal_modules(vae_dim: int):
    """Temporal position projection + per-frame decoder (shared across projectors)."""
    temporal_pos_proj = nn.Sequential(
        nn.Linear(vae_dim, vae_dim), nn.SiLU(), nn.Linear(vae_dim, vae_dim),
    )
    frame_decoder = nn.Sequential(
        nn.Linear(vae_dim, vae_dim * 2), nn.GELU(), nn.Linear(vae_dim * 2, vae_dim),
    )
    return temporal_pos_proj, frame_decoder


# ────────────────────────────────────────────────────────────────────────────
# 1. Default projector (backward-compatible, single-frame / mean-collapsed)
# ────────────────────────────────────────────────────────────────────────────

class VideoAlignProjector(nn.Module):
    """Per-patch cosine alignment with temporally-averaged VAE features."""

    multi_frame = False

    def __init__(self, vla_dim: int, vae_dim: int = 48, use_vla_norm: bool = False):
        super().__init__()
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        _init_weights(self)

    def project_vla(self, h: torch.Tensor) -> torch.Tensor:
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        vla_proj = F.normalize(self.project_vla(vla_hidden), dim=-1)
        vae_norm = F.normalize(vae_hidden, dim=-1)
        cos_sim = (vla_proj * vae_norm).sum(dim=-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 2. Multi-frame cosine (max / softmax)
# ────────────────────────────────────────────────────────────────────────────

class MultiFrameCosineProjector(nn.Module):
    """Per-patch cosine against every temporal frame, aggregated via max or softmax."""

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 temporal_pool: str = "softmax", softmax_temperature: float = 0.1,
                 use_vla_norm: bool = False):
        super().__init__()
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.temporal_pool = temporal_pool
        self.softmax_temperature = softmax_temperature
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        vla_proj = F.normalize(self.project_vla(vla_hidden), dim=-1)
        vae_norm = F.normalize(vae_hidden, dim=-1)
        cos_sim = torch.einsum("bnd,bntd->bnt", vla_proj, vae_norm)
        if self.temporal_pool == "max":
            cos_agg = cos_sim.max(dim=-1).values
        elif self.temporal_pool == "softmax":
            w = F.softmax(cos_sim / self.softmax_temperature, dim=-1)
            cos_agg = (w * cos_sim).sum(dim=-1)
        else:
            raise ValueError(self.temporal_pool)
        return _masked_cosine_loss(cos_agg, mask)


# ────────────────────────────────────────────────────────────────────────────
# 3. Temporal Reconstruction  (MLP decoder, blind)
# ────────────────────────────────────────────────────────────────────────────

class TemporalReconstructAlignProjector(nn.Module):
    """Blind per-frame reconstruction via position-conditioned MLP decoder.

    ``predicted[n, t] = Decoder(VLA_proj[n] + pos(t))``

    Prediction sees **no** real VAE features.  T'× supervision.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48, use_vla_norm: bool = False):
        super().__init__()
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.temporal_pos_proj, self.frame_decoder = _make_temporal_modules(vae_dim)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = self.project_vla(vla_hidden)
        t_cue = self.temporal_pos_proj(_sinusoidal_temporal_embed(T, C, vla_proj.device))
        predicted = self.frame_decoder(vla_proj.unsqueeze(2) + t_cue)
        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 3b. Multi-Frame Concat  (output T'*C, no positional encoding)
# ────────────────────────────────────────────────────────────────────────────

class MultiFrameConcatAlignProjector(nn.Module):
    """Per-frame reconstruction by projecting directly to T'×C.

    ``VLA_proj(VLA_hidden[n]) → [T'*C] → reshape [T', C]``

    Same two-layer MLP as the mean projector, but with a wider output
    (num_frames × vae_dim instead of vae_dim).  No extra decoder layer,
    no bottleneck, no positional encoding.

    Prediction sees **no** real VAE features.  T'× supervision.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 2, use_vla_norm: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.vae_dim = vae_dim
        out_dim = num_frames * vae_dim
        self.fc1 = nn.Linear(vla_dim, out_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim * 4, out_dim)
        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        predicted = self.project_vla(vla_hidden).reshape(B, N, self.num_frames, C)
        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)
        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 3b'. Multi-Frame Concat with VLA-anchored head (for high-dim targets like DiT)
# ────────────────────────────────────────────────────────────────────────────

class MultiFrameConcatVlaAnchoredProjector(nn.Module):
    """Same forward as ``MultiFrameConcatAlignProjector``, but head dim is anchored
    to ``vla_dim`` instead of ``out_dim``.

    The original projector uses ``hidden = 4 × out_dim``, which works for small
    targets (VAE: out_dim=96 → hidden=384, head ≈ 0.4M) but explodes for large
    targets (DiT: out_dim=6144 → hidden=24576, head ≈ 200M).  When the head
    becomes overparameterized relative to the alignment task, it can fit the
    target by itself ("head absorption"), leaving the VLA backbone under-trained.

    By anchoring hidden to ``vla_dim``, head size stays ~12-16M regardless of
    target dim, forcing the backbone to actually encode target-aligned features.

    Two head styles are supported:

    - ``head_style="vla_anchored"`` (default):
        ``vla_dim → vla_dim → out_dim`` (2-layer MLP, ~16M params for vla_dim=2048)
    - ``head_style="linear"``:
        ``vla_dim → out_dim`` (single linear, ~12M params, no nonlinearity)

    Forward / loss computation is identical to ``MultiFrameConcatAlignProjector``.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 2, use_vla_norm: bool = False,
                 head_style: str = "vla_anchored"):
        super().__init__()
        if head_style not in ("vla_anchored", "linear"):
            raise ValueError(
                f"head_style must be 'vla_anchored' or 'linear', got {head_style!r}"
            )

        self.num_frames = num_frames
        self.vae_dim = vae_dim
        self.head_style = head_style
        out_dim = num_frames * vae_dim

        if head_style == "linear":
            self.fc1 = nn.Linear(vla_dim, out_dim)
            self.act = nn.Identity()
            self.fc2 = nn.Identity()
        else:  # "vla_anchored"
            self.fc1 = nn.Linear(vla_dim, vla_dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(vla_dim, out_dim)

        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        predicted = self.project_vla(vla_hidden).reshape(B, N, self.num_frames, C)
        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)
        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 3c. Dual-Head Concat  (two independent heads, separate fitting)
# ────────────────────────────────────────────────────────────────────────────

class DualHeadConcatAlignProjector(nn.Module):
    """Two independent MLP heads that separately fit primary and auxiliary VAE features.

    Architecture per head is identical to ``MultiFrameConcatAlignProjector``:
    ``VLA_proj(VLA_hidden[n]) → [T'×C] → reshape [T', C]``.

    When ``vae_hidden_aux`` is ``None`` the auxiliary head is skipped and the
    module behaves exactly like a single ``MultiFrameConcatAlignProjector``,
    so old single-cache configs keep working.
    """

    multi_frame = True
    dual_head = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 2, num_frames_aux: int = 2,
                 use_vla_norm: bool = False,
                 loss_weight_primary: float = 1.0,
                 loss_weight_aux: float = 1.0):
        super().__init__()
        self.num_frames = num_frames
        self.num_frames_aux = num_frames_aux
        self.vae_dim = vae_dim
        self.loss_weight_primary = loss_weight_primary
        self.loss_weight_aux = loss_weight_aux

        out_p = num_frames * vae_dim
        self.fc1 = nn.Linear(vla_dim, out_p * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_p * 4, out_p)

        out_a = num_frames_aux * vae_dim
        self.fc1_aux = nn.Linear(vla_dim, out_a * 4)
        self.act_aux = nn.GELU()
        self.fc2_aux = nn.Linear(out_a * 4, out_a)

        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def project_vla_aux(self, h):
        return _project_vla(h, self.fc1_aux, self.act_aux, self.fc2_aux, self.vla_norm)

    @staticmethod
    def _head_loss(predicted, target, num_frames, mask):
        B, N, T, C = target.shape
        predicted = predicted.reshape(B, N, num_frames, C)
        if num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)
        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(target, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)

    def forward(self, vla_hidden, vae_hidden, mask=None, vae_hidden_aux=None):
        pred_p = self.project_vla(vla_hidden)
        loss_p = self._head_loss(pred_p, vae_hidden, self.num_frames, mask)

        if vae_hidden_aux is None:
            return loss_p

        pred_a = self.project_vla_aux(vla_hidden)
        loss_a = self._head_loss(pred_a, vae_hidden_aux, self.num_frames_aux, mask)

        w_p, w_a = self.loss_weight_primary, self.loss_weight_aux
        return (w_p * loss_p + w_a * loss_a) / (w_p + w_a)


# ────────────────────────────────────────────────────────────────────────────
# Ablation A: multi_frame_concat + spatial PE on VAE target
# ────────────────────────────────────────────────────────────────────────────

class MultiFrameSpatialPEProjector(nn.Module):
    """multi_frame_concat + spatial PE on VAE target (Spatial-Forcing style).

    Identical MLP as ``multi_frame_concat``.  The only addition: 2-D sinusoidal
    spatial PE is added to VAE target features before loss computation, so that
    the loss is position-aware even after spatial downsampling.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 2, spatial_pe_ratio: float = 0.1,
                 use_vla_norm: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.vae_dim = vae_dim
        self.spatial_pe_ratio = spatial_pe_ratio
        out_dim = num_frames * vae_dim
        self.fc1 = nn.Linear(vla_dim, out_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim * 4, out_dim)
        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def _add_spatial_pe(self, x: torch.Tensor) -> torch.Tensor:
        """Add 2-D PE to [B, N, D] features (per-camera)."""
        B, N, D = x.shape
        G, H, W = _decompose_cameras(N)
        x_2d = x.reshape(B * G, H * W, D).permute(0, 2, 1).reshape(B * G, D, H, W)
        x_2d = apply_spatial_pe(x_2d, ratio=self.spatial_pe_ratio)
        return x_2d.flatten(2).permute(0, 2, 1).reshape(B, N, D)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        predicted = self.project_vla(vla_hidden).reshape(B, N, self.num_frames, C)
        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)
        vae_with_pe = torch.stack(
            [self._add_spatial_pe(vae_hidden[:, :, t, :]) for t in range(T)], dim=2,
        )
        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_with_pe, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# Ablation B: multi_frame_concat + temporal PE on VAE target
# ────────────────────────────────────────────────────────────────────────────

class MultiFrameTemporalPEProjector(nn.Module):
    """multi_frame_concat + sinusoidal temporal PE on VAE target.

    Identical MLP as ``multi_frame_concat``.  The only addition: sinusoidal
    temporal positional encoding is added to VAE target features before loss
    computation, giving each frame a distinct temporal identity (frame 0 ≠
    frame 1 ≠ frame 2) so the loss can distinguish temporal positions.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 2, temporal_pe_ratio: float = 0.1,
                 use_vla_norm: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.vae_dim = vae_dim
        self.temporal_pe_ratio = temporal_pe_ratio
        out_dim = num_frames * vae_dim
        self.fc1 = nn.Linear(vla_dim, out_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim * 4, out_dim)
        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        predicted = self.project_vla(vla_hidden).reshape(B, N, self.num_frames, C)
        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)
        t_pe = _sinusoidal_temporal_embed(T, C, vae_hidden.device)  # [T, C]
        vae_with_pe = vae_hidden + self.temporal_pe_ratio * t_pe
        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_with_pe, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# Ablation A2: multi_frame_concat + spatial PE on VLA input
# ────────────────────────────────────────────────────────────────────────────

class MultiFrameVLASpatialPEProjector(nn.Module):
    """multi_frame_concat + spatial PE on VLA input before MLP.

    Identical MLP as ``multi_frame_concat``.  The only addition: 2-D sinusoidal
    spatial PE is added to VLA hidden states **before** the MLP projection,
    so the MLP can use absolute spatial position during projection.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 2, spatial_pe_ratio: float = 0.1,
                 use_vla_norm: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.vae_dim = vae_dim
        self.spatial_pe_ratio = spatial_pe_ratio
        out_dim = num_frames * vae_dim
        self.fc1 = nn.Linear(vla_dim, out_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim * 4, out_dim)
        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def _add_spatial_pe(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        G, H, W = _decompose_cameras(N)
        x_2d = x.reshape(B * G, H * W, D).permute(0, 2, 1).reshape(B * G, D, H, W)
        x_2d = apply_spatial_pe(x_2d, ratio=self.spatial_pe_ratio)
        return x_2d.flatten(2).permute(0, 2, 1).reshape(B, N, D)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_with_pe = self._add_spatial_pe(vla_hidden)
        predicted = self.project_vla(vla_with_pe).reshape(B, N, self.num_frames, C)
        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)
        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# Ablation B2: multi_frame_concat + temporal PE + refine MLP on VLA output
# ────────────────────────────────────────────────────────────────────────────

class MultiFrameVLATemporalPERefineProjector(nn.Module):
    """multi_frame_concat + temporal PE injected between two MLP stages.

    MLP1 directly outputs T'×C (same as multi_frame_concat, no bottleneck).
    After reshaping to per-frame features, sinusoidal temporal PE is added,
    then a lightweight MLP2 refines each frame conditioned on the temporal
    signal.  MLP2 sees "which frame am I" and adjusts accordingly.

    Unlike ``temporal_reconstruct`` there is **no bottleneck** — MLP1 already
    gives each frame its own dedicated features; MLP2 only does residual
    correction informed by temporal position.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 2, use_vla_norm: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.vae_dim = vae_dim
        out_dim = num_frames * vae_dim
        self.fc1 = nn.Linear(vla_dim, out_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim * 4, out_dim)
        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        self.refine = nn.Sequential(
            nn.Linear(vae_dim, vae_dim * 2), nn.GELU(), nn.Linear(vae_dim * 2, vae_dim),
        )
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        predicted = self.project_vla(vla_hidden).reshape(B, N, self.num_frames, C)
        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)
        t_pe = _sinusoidal_temporal_embed(T, C, predicted.device)  # [T, C]
        predicted = predicted + self.refine(predicted + t_pe)
        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)
# ────────────────────────────────────────────────────────────────────────────

class ConvDirectMultiFrameProjector(nn.Module):
    """multi_frame_concat with Conv2d replacing MLP (iREPA-style, matched capacity).

    Same 4× expansion as ``multi_frame_concat``, but using Conv2d layers
    instead of Linear.  Conv1×1 for channel projection (equivalent to Linear)
    + Conv3×3 for spatial mixing.  No spatial PE, no temporal conv, no spatial
    norm — purely tests whether conv's spatial locality improves over MLP.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 2, num_cameras: int = 1,
                 use_vla_norm: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        self.vae_dim = vae_dim
        out_dim = num_frames * vae_dim
        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        self.conv1 = nn.Conv2d(num_cameras * vla_dim, out_dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_dim * 4, num_cameras * out_dim, kernel_size=3, padding=1)
        _init_weights(self)

    def _camera_grid(self, N):
        G = self.num_cameras
        side = int((N // G) ** 0.5)
        return G, side, side

    def project_vla(self, h):
        if self.vla_norm is not None:
            h = self.vla_norm(h)
        B, N, D = h.shape
        G, H, W = self._camera_grid(N)
        x = h.reshape(B, G, H, W, D).permute(0, 4, 1, 2, 3).reshape(B, G * D, H, W)
        x = self.conv2(self.act(self.conv1(x)))                    # [B, G*nf*C, H, W]
        x = x.reshape(B, G, self.num_frames, self.vae_dim, H, W)
        x = x.permute(0, 1, 4, 5, 2, 3).reshape(B, N, self.num_frames, self.vae_dim)
        return x.mean(dim=2)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        G, H, W = self._camera_grid(N)
        h = vla_hidden
        if self.vla_norm is not None:
            h = self.vla_norm(h)
        D = h.shape[-1]
        x = h.reshape(B, G, H, W, D).permute(0, 4, 1, 2, 3).reshape(B, G * D, H, W)
        x = self.conv2(self.act(self.conv1(x)))                    # [B, G*nf*C, H, W]
        x = x.reshape(B, G, self.num_frames, C, H, W)
        predicted = x.permute(0, 1, 4, 5, 2, 3).reshape(B, N, self.num_frames, C)
        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)
        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 4. Temporal Flow  (predict inter-frame dynamics, blind)
# ────────────────────────────────────────────────────────────────────────────

class TemporalFlowAlignProjector(nn.Module):
    """Predict inter-frame *feature dynamics* Δ_t = VAE[t+1] − VAE[t].

    **Motivation**: for robot control, *motion* matters more than static
    appearance.  Predicting temporal differences directly forces the VLA
    token to encode the motion trajectory, not just a static summary.

    Loss = ``static_weight × L_static + (1 − static_weight) × L_flow``

    - ``L_static``: cosine alignment between VLA_proj and temporal mean of VAE
      (anchors basic spatial correspondence).
    - ``L_flow``: per-step cosine alignment between predicted and actual Δ_t
      (forces encoding of dynamics).

    Prediction sees **no** real VAE features.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 static_weight: float = 0.3, use_vla_norm: bool = False):
        super().__init__()
        self.static_weight = static_weight
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.temporal_pos_proj, self.flow_decoder = _make_temporal_modules(vae_dim)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        """vla_hidden [B,N,D_vla], vae_hidden [B,N,T',D_vae], mask [B,N]."""
        B, N, T, C = vae_hidden.shape
        vla_proj = self.project_vla(vla_hidden)

        # ── static anchor ────────────────────────────────────────────────
        vla_n = F.normalize(vla_proj, dim=-1)
        static_cos = (vla_n * F.normalize(vae_hidden.mean(2), dim=-1)).sum(-1)  # [B, N]

        # ── flow prediction ──────────────────────────────────────────────
        actual_flow = vae_hidden[:, :, 1:] - vae_hidden[:, :, :-1]             # [B, N, T-1, C]
        T_flow = T - 1

        if T_flow > 0:
            t_cue = self.temporal_pos_proj(_sinusoidal_temporal_embed(T_flow, C, vla_proj.device))
            predicted_flow = self.flow_decoder(vla_proj.unsqueeze(2) + t_cue)
            flow_cos = (F.normalize(predicted_flow, dim=-1)
                        * F.normalize(actual_flow, dim=-1)).sum(-1).mean(-1)   # [B, N]
        else:
            flow_cos = static_cos

        cos_sim = self.static_weight * static_cos + (1.0 - self.static_weight) * flow_cos
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 5. Autoregressive GRU  (sequential prediction, blind)
# ────────────────────────────────────────────────────────────────────────────

class AutoregressiveAlignProjector(nn.Module):
    """Autoregressive frame prediction from VLA hidden state via GRU.

    The VLA projection initialises a GRU hidden state.  At each step the GRU
    receives a temporal position cue and emits a predicted frame feature.

    **Why this is harder**: errors accumulate through the recurrence, so the
    VLA token must encode **precise** trajectory information for all future
    frames — not just a rough summary.

    Prediction sees **no** real VAE features.  T'× supervision.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48, use_vla_norm: bool = False):
        super().__init__()
        self.vae_dim = vae_dim
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.temporal_pos_proj = nn.Sequential(
            nn.Linear(vae_dim, vae_dim), nn.SiLU(), nn.Linear(vae_dim, vae_dim),
        )
        self.gru = nn.GRUCell(vae_dim, vae_dim)
        self.frame_head = nn.Sequential(
            nn.Linear(vae_dim, vae_dim * 2), nn.GELU(), nn.Linear(vae_dim * 2, vae_dim),
        )
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = self.project_vla(vla_hidden)
        t_cue = self.temporal_pos_proj(_sinusoidal_temporal_embed(T, C, vla_proj.device))  # [T, C]

        h = vla_proj.reshape(B * N, C)
        preds = []
        for t in range(T):
            inp = t_cue[t].unsqueeze(0).expand(B * N, -1)
            h = self.gru(inp, h)
            preds.append(self.frame_head(h))

        predicted = torch.stack(preds, dim=1).reshape(B, N, T, C)
        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 6. Variational Temporal Bottleneck  (VAE-style, blind)
# ────────────────────────────────────────────────────────────────────────────

class VariationalTemporalAlignProjector(nn.Module):
    """VAE-style information bottleneck for temporal alignment.

    ``VLA → μ, log σ² → z ~ N(μ, σ²) → z + pos(t) → Decoder → predicted[t]``

    **Motivation**: the KL regulariser ``KL(q(z|VLA) ‖ N(0,I))`` forces the
    latent code to be a *smooth, structured* representation of the temporal
    trajectory — rather than just memorising a lookup table per position.
    β controls compression vs. reconstruction fidelity (β-VAE trade-off).

    Prediction sees **no** real VAE features.  T'× supervision.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 kl_weight: float = 1e-2, use_vla_norm: bool = False):
        super().__init__()
        self.kl_weight = kl_weight
        self.vae_dim = vae_dim
        # Encoder: VLA → hidden → μ, logσ²
        self.enc_fc = nn.Linear(vla_dim, vae_dim * 4)
        self.enc_act = nn.GELU()
        self.fc_mu = nn.Linear(vae_dim * 4, vae_dim)
        self.fc_logvar = nn.Linear(vae_dim * 4, vae_dim)
        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        # Decoder
        self.temporal_pos_proj, self.frame_decoder = _make_temporal_modules(vae_dim)
        _init_weights(self)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape

        if self.vla_norm is not None:
            vla_hidden = self.vla_norm(vla_hidden)
        h = self.enc_act(self.enc_fc(vla_hidden))                      # [B, N, C*4]
        mu = self.fc_mu(h)                                             # [B, N, C]
        logvar = self.fc_logvar(h)                                     # [B, N, C]

        # Reparameterise (no sampling at eval → use mu)
        if self.training:
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z = mu

        # Decode each frame
        t_cue = self.temporal_pos_proj(_sinusoidal_temporal_embed(T, C, z.device))
        predicted = self.frame_decoder(z.unsqueeze(2) + t_cue)         # [B, N, T, C]

        # Reconstruction loss
        cos_sim = (F.normalize(predicted, dim=-1)
                   * F.normalize(vae_hidden, dim=-1)).sum(-1).mean(-1) # [B, N]
        if mask is not None:
            m = mask.float()
            recon = (1.0 - cos_sim * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            recon = (1.0 - cos_sim).mean(-1)

        # KL divergence: KL(N(μ, σ²) ‖ N(0, I))
        kl_per_token = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(-1)  # [B, N]
        if mask is not None:
            kl = (kl_per_token * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            kl = kl_per_token.mean(-1)

        return (recon + self.kl_weight * kl).mean()


# ────────────────────────────────────────────────────────────────────────────
# 7. Contrastive Temporal Alignment  (InfoNCE, blind)
# ────────────────────────────────────────────────────────────────────────────

class ContrastiveTemporalAlignProjector(nn.Module):
    """InfoNCE contrastive alignment across spatial locations.

    For each VLA patch *n*, the T' temporal VAE features at that location are
    **positives** and the T' features at every *other* spatial location are
    **negatives**.  The VLA projection must be *discriminatively* close to its
    own temporal features — not just generically similar.

    ``L = −log  Σ_{t} exp(sim(q_n, k_{n,t}) / τ)
                ─────────────────────────────────
                Σ_{m,t} exp(sim(q_n, k_{m,t}) / τ)``

    **Why this is stronger**: cosine alignment only pushes for high absolute
    similarity.  InfoNCE additionally pushes *away* from negatives, forcing
    each VLA token to encode location-specific, temporally-rich information
    that distinguishes it from other patches.

    Prediction sees **no** real VAE features during projection.  T'× rich
    positive set.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 temperature: float = 0.07, use_vla_norm: bool = False):
        super().__init__()
        self.temperature = temperature
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        """vla_hidden [B,N,D_vla], vae_hidden [B,N,T',D_vae], mask [B,N]."""
        B, N, T, C = vae_hidden.shape
        q = F.normalize(self.project_vla(vla_hidden), dim=-1)          # [B, N, C]
        k = F.normalize(vae_hidden.reshape(B, N * T, C), dim=-1)      # [B, N*T', C]

        # Similarity matrix: [B, N, N*T']
        logits = torch.bmm(q, k.transpose(1, 2)) / self.temperature

        # Positive mask: for query n, positives are indices [n*T .. n*T+T-1]
        pos_idx = torch.arange(N, device=logits.device)
        pos_start = (pos_idx * T).unsqueeze(1).expand(-1, T)
        pos_offsets = torch.arange(T, device=logits.device).unsqueeze(0).expand(N, -1)
        pos_flat = (pos_start + pos_offsets).reshape(N, T)             # [N, T]

        # Multi-positive InfoNCE: −log( Σ_pos exp / Σ_all exp )
        log_sum_all = torch.logsumexp(logits, dim=-1)                  # [B, N]
        # Gather positive logits
        pos_flat_exp = pos_flat.unsqueeze(0).expand(B, -1, -1)         # [B, N, T]
        pos_logits = logits.gather(2, pos_flat_exp)                    # [B, N, T]
        log_sum_pos = torch.logsumexp(pos_logits, dim=-1)              # [B, N]

        per_token_loss = log_sum_all - log_sum_pos                     # [B, N]

        if mask is not None:
            m = mask.float()
            loss = (per_token_loss * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            loss = per_token_loss.mean(-1)
        return loss.mean()


# ────────────────────────────────────────────────────────────────────────────
# 8. Cross-Attention Reconstruction (has VAE KV access)
# ────────────────────────────────────────────────────────────────────────────

class CrossAttentionReconstructProjector(nn.Module):
    """Cross-attention temporal reconstruction.

    Builds T' temporal queries from VLA_proj + pos(t), cross-attends to VAE
    key/values, reconstructs per-frame targets.

    **Note**: unlike the blind projectors above, queries *can* retrieve
    information from real VAE features via cross-attention.  This is a
    stronger architecture but a weaker inductive constraint.  Use this to
    ablate whether blind prediction or richer architecture matters more.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_heads: int = 4, use_vla_norm: bool = False):
        super().__init__()
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.temporal_pos_proj = nn.Sequential(
            nn.Linear(vae_dim, vae_dim), nn.SiLU(), nn.Linear(vae_dim, vae_dim),
        )
        self.cross_attn = nn.MultiheadAttention(vae_dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(vae_dim)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = self.project_vla(vla_hidden)
        t_cue = self.temporal_pos_proj(_sinusoidal_temporal_embed(T, C, vla_proj.device))
        queries = vla_proj.unsqueeze(2) + t_cue

        q = queries.reshape(B * N, T, C)
        kv = vae_hidden.reshape(B * N, T, C)
        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = self.out_norm(attn_out).reshape(B, N, T, C)

        cos_sim = (F.normalize(attn_out, dim=-1) * F.normalize(vae_hidden, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 9. Current-Frame-Aware Reconstruction  (signed delta encoding, blind)
# ────────────────────────────────────────────────────────────────────────────

class CurrentFrameAwareAlignProjector(nn.Module):
    """Temporal reconstruction with **signed representative-delta encoding**
    and explicit current/past/future loss decomposition.

    The video VAE compresses with a 4n+1 pattern (1st frame independent,
    then groups of 4).  This projector computes a **representative delta**
    for each VAE temporal feature — the mean of the frame deltas it covers —
    and uses that as the signed position encoding.

    The VAE feature whose representative delta is closest to 0 is treated as
    the approximate "current frame".  When ``video_delta_frames`` starts
    with 0 the match is exact; otherwise it is the best available proxy.

    Loss decomposes into three parts:

    - **current** (repr_δ closest to 0): anchor alignment.
    - **future** (repr_δ > 0): forward dynamics prediction.
    - **past** (repr_δ < 0): history reconstruction.

    Default weights emphasise future prediction.
    Prediction sees **no** real VAE features.  T'× supervision.
    """

    multi_frame = True

    def __init__(
        self,
        vla_dim: int,
        vae_dim: int = 48,
        video_delta_frames: list[int] | tuple[int, ...] | None = None,
        weight_current: float = 0.2,
        weight_future: float = 0.5,
        weight_past: float = 0.3,
        use_vla_norm: bool = False,
    ):
        super().__init__()
        repr_deltas = _compute_vae_representative_deltas(video_delta_frames or [0])
        self.register_buffer("repr_deltas", torch.tensor(repr_deltas, dtype=torch.float32))
        # Index of the VAE feature closest to δ=0 (approximate current frame)
        self.current_idx = int(torch.tensor(repr_deltas).abs().argmin().item())
        self.weight_current = weight_current
        self.weight_future = weight_future
        self.weight_past = weight_past
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.temporal_pos_proj, self.frame_decoder = _make_temporal_modules(vae_dim)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = self.project_vla(vla_hidden)

        deltas = self.repr_deltas[:T].to(vla_proj.device)
        t_cue = self.temporal_pos_proj(_sinusoidal_delta_embed(deltas, C))
        predicted = self.frame_decoder(vla_proj.unsqueeze(2) + t_cue)  # [B, N, T, C]

        per_frame = (F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)).sum(-1)  # [B, N, T]

        parts, weights = [], []
        cur_idx = min(self.current_idx, T - 1)
        cur_set = {cur_idx}
        fut_idx = [i for i in range(T) if i not in cur_set and deltas[i] > 0]
        past_idx = [i for i in range(T) if i not in cur_set and deltas[i] < 0]

        parts.append(per_frame[:, :, cur_idx])
        weights.append(self.weight_current)
        if fut_idx:
            parts.append(per_frame[:, :, fut_idx].mean(-1))
            weights.append(self.weight_future)
        if past_idx:
            parts.append(per_frame[:, :, past_idx].mean(-1))
            weights.append(self.weight_past)

        w_total = sum(weights)
        cos_sim = sum(w * p for w, p in zip(weights, parts)) / w_total
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 10. Causal Prediction  (signed delta + current-first + causal weighting, blind)
# ────────────────────────────────────────────────────────────────────────────

class CausalPredictionAlignProjector(nn.Module):
    """Causal temporal prediction centred on the current frame.

    Structurally distinct from ``TemporalReconstructAlignProjector`` in
    three ways:

    1. **Signed representative-delta** position encoding (from the 4n+1 VAE
       pattern) instead of 0..T-1 indices — the decoder knows which
       direction (past/future) and how far from current.
    2. **Current-first anchor**: the VAE feature closest to δ=0 is treated
       as the VLA's own observation; the model expands outward from it.
    3. **Causal loss weighting**: future > current > past.

    Decoder is a simple MLP (same as temporal_reconstruct), keeping the
    ablation clean — differences come purely from the temporal structure.

    Prediction sees **no** real VAE features.  T'× supervision.
    """

    multi_frame = True

    def __init__(
        self,
        vla_dim: int,
        vae_dim: int = 48,
        video_delta_frames: list[int] | tuple[int, ...] | None = None,
        weight_current: float = 0.3,
        weight_future: float = 0.5,
        weight_past: float = 0.2,
        use_vla_norm: bool = False,
    ):
        super().__init__()
        repr_deltas = _compute_vae_representative_deltas(video_delta_frames or [0])
        self.register_buffer("repr_deltas", torch.tensor(repr_deltas, dtype=torch.float32))
        self.current_idx = int(torch.tensor(repr_deltas).abs().argmin().item())
        self.weight_current = weight_current
        self.weight_future = weight_future
        self.weight_past = weight_past

        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.temporal_pos_proj, self.frame_decoder = _make_temporal_modules(vae_dim)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = self.project_vla(vla_hidden)

        deltas = self.repr_deltas[:T].to(vla_proj.device)
        t_cue = self.temporal_pos_proj(_sinusoidal_delta_embed(deltas, C))
        predicted = self.frame_decoder(vla_proj.unsqueeze(2) + t_cue)  # [B, N, T, C]

        per_frame = (F.normalize(predicted, dim=-1)
                     * F.normalize(vae_hidden, dim=-1)).sum(-1)        # [B, N, T]

        cur_idx = min(self.current_idx, T - 1)
        cur_set = {cur_idx}
        fut_idx = [i for i in range(T) if i not in cur_set and deltas[i] > 0]
        past_idx = [i for i in range(T) if i not in cur_set and deltas[i] < 0]

        parts, weights = [], []
        parts.append(per_frame[:, :, cur_idx])
        weights.append(self.weight_current)
        if fut_idx:
            parts.append(per_frame[:, :, fut_idx].mean(-1))
            weights.append(self.weight_future)
        if past_idx:
            parts.append(per_frame[:, :, past_idx].mean(-1))
            weights.append(self.weight_past)

        w_total = sum(weights)
        cos_sim = sum(w * p for w, p in zip(weights, parts)) / w_total
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 11. Dual-head: separate current + future alignment
# ────────────────────────────────────────────────────────────────────────────

class DualHeadAlignProjector(nn.Module):
    """Two independent MLPs: one aligns with the current frame (T'=0),
    the other predicts the mean of future frames (T'>0).

    Both gradients flow back to VLA backbone independently — no
    interference in the projector parameters.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 future_weight: float = 1.0, use_vla_norm: bool = False):
        super().__init__()
        self.future_weight = future_weight
        self.cur_fc1, self.cur_act, self.cur_fc2, self.cur_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.fut_fc1, self.fut_act, self.fut_fc2, self.fut_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.cur_fc1, self.cur_act, self.cur_fc2, self.cur_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        # current frame
        cur_proj = F.normalize(_project_vla(vla_hidden, self.cur_fc1, self.cur_act, self.cur_fc2, self.cur_norm), dim=-1)
        cur_cos = (cur_proj * F.normalize(vae_hidden[:, :, 0], dim=-1)).sum(-1)
        loss_cur = _masked_cosine_loss(cur_cos, mask)

        if T <= 1:
            return loss_cur

        # future frames (mean of T'>0)
        fut_proj = F.normalize(_project_vla(vla_hidden, self.fut_fc1, self.fut_act, self.fut_fc2, self.fut_norm), dim=-1)
        fut_target = F.normalize(vae_hidden[:, :, 1:].mean(2), dim=-1)
        fut_cos = (fut_proj * fut_target).sum(-1)
        loss_fut = _masked_cosine_loss(fut_cos, mask)

        # return loss_cur + self.future_weight * loss_fut
        return (loss_cur + loss_fut) / 2.0


# ────────────────────────────────────────────────────────────────────────────
# 12. Temporal Decay Cosine  (fixed decay weighting, zero new params)
# ────────────────────────────────────────────────────────────────────────────

class TemporalDecayCosineProjector(nn.Module):
    """Per-frame cosine with exponential-decay temporal weighting.

    Same MLP as ``mean``, but computes per-frame cosine similarities and
    weights them by ``exp(-α·t)`` so the current frame (t=0) dominates
    while future frames contribute with geometrically decreasing influence.

    Unlike ``softmax_cosine`` whose weights are *data-dependent* (the model
    attends to whichever frame it already matches, creating a self-reinforcing
    loop), decay weights are **fixed** — providing a stable inductive bias
    that current observations matter most for action prediction.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 decay_alpha: float = 0.5, use_vla_norm: bool = False):
        super().__init__()
        self.decay_alpha = decay_alpha
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = F.normalize(self.project_vla(vla_hidden), dim=-1)  # [B, N, C]
        vae_norm = F.normalize(vae_hidden, dim=-1)                    # [B, N, T, C]

        per_frame_cos = (vla_proj.unsqueeze(2) * vae_norm).sum(-1)    # [B, N, T]

        t_idx = torch.arange(T, device=per_frame_cos.device, dtype=torch.float32)
        weights = torch.exp(-self.decay_alpha * t_idx)
        weights = weights / weights.sum()

        cos_sim = (per_frame_cos * weights.view(1, 1, T)).sum(-1)     # [B, N]
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 13. Mean + Dynamics Direction  (mean anchor + motion direction auxiliary)
# ────────────────────────────────────────────────────────────────────────────

class MeanDynamicsProjector(nn.Module):
    """Mean cosine alignment augmented with temporal dynamics prediction.

    Primary: ``cosine(VLA_proj, temporal_mean(VAE))`` — identical to the
    ``mean`` projector, ensuring optimisation stability.

    Auxiliary: predict the **direction** of temporal change::

        target = normalize(mean(VAE[1:]) − VAE[0])
        pred   = normalize(dynamics_head(VLA_proj))
        loss   = 1 − cosine(pred, target)

    The dynamics target is a *single normalised vector* (same dimensionality
    as the mean target), so it is far easier to predict than per-frame
    reconstruction.  It captures **how** features are changing — motion
    direction — which is directly action-relevant for robot control.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 dynamics_weight: float = 0.3, use_vla_norm: bool = False):
        super().__init__()
        self.dynamics_weight = dynamics_weight
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.dynamics_head = nn.Linear(vae_dim, vae_dim)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = self.project_vla(vla_hidden)

        # ── primary: mean cosine (same as VideoAlignProjector) ────
        vae_mean = vae_hidden.mean(dim=2)
        vla_n = F.normalize(vla_proj, dim=-1)
        cos_sim = (vla_n * F.normalize(vae_mean, dim=-1)).sum(-1)
        primary_loss = _masked_cosine_loss(cos_sim, mask)

        if T <= 1:
            return primary_loss

        # ── auxiliary: temporal dynamics direction ─────────────────
        temporal_dir = F.normalize(
            vae_hidden[:, :, 1:].mean(2) - vae_hidden[:, :, 0], dim=-1,
        )
        pred_dir = F.normalize(self.dynamics_head(vla_proj), dim=-1)
        dir_cos = (pred_dir * temporal_dir).sum(-1)
        dynamics_loss = _masked_cosine_loss(dir_cos, mask)

        return primary_loss + self.dynamics_weight * dynamics_loss


# ────────────────────────────────────────────────────────────────────────────
# 14. Multi-Scale Temporal Mean  (hierarchical temporal cosine, zero new params)
# ────────────────────────────────────────────────────────────────────────────

class MultiScaleTemporalMeanProjector(nn.Module):
    """Hierarchical multi-scale temporal cosine alignment.

    A single VLA projection is cosine-aligned with **three** temporal
    scales simultaneously:

    - **current** (t=0): precise instantaneous observation.
    - **near**    (t=0..T'//2): short-horizon context.
    - **full**    (t=0..T'-1): trajectory-level summary.

    When the scene is static, these three targets coincide and the loss
    behaves identically to ``mean``.  When the scene changes rapidly,
    the targets diverge — and the optimal VLA projection must *interpolate*
    among them, naturally encoding both state and dynamics without
    explicit per-frame reconstruction.

    Zero extra parameters beyond the standard MLP.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 weight_current: float = 0.5, weight_near: float = 0.3,
                 weight_full: float = 0.2, use_vla_norm: bool = False):
        super().__init__()
        self.w_cur = weight_current
        self.w_near = weight_near
        self.w_full = weight_full
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = F.normalize(self.project_vla(vla_hidden), dim=-1)

        cos_cur = (vla_proj * F.normalize(vae_hidden[:, :, 0], dim=-1)).sum(-1)
        cos_full = (vla_proj * F.normalize(vae_hidden.mean(2), dim=-1)).sum(-1)

        if T > 2:
            mid = max(T // 2, 1)
            cos_near = (vla_proj * F.normalize(vae_hidden[:, :, :mid].mean(2), dim=-1)).sum(-1)
            w_total = self.w_cur + self.w_near + self.w_full
            cos_sim = (self.w_cur * cos_cur + self.w_near * cos_near + self.w_full * cos_full) / w_total
        else:
            w_total = self.w_cur + self.w_near + self.w_full
            cos_sim = ((self.w_cur + self.w_near) * cos_cur + self.w_full * cos_full) / w_total

        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# 15. Residual Temporal — mean anchor  (proven spatial + temporal decomposition)
# ────────────────────────────────────────────────────────────────────────────

class ResidualTemporalMeanProjector(nn.Module):
    """Mean cosine + temporal-residual L2, anchored on temporal mean.

    - **Spatial**: ``cosine(VLA_proj, mean(VAE))`` — identical to ``mean``.
    - **Temporal**: ``L2(decoder(VLA_proj + pos(t)), VAE[t] − mean(VAE))``
      for all t — predicts each frame's deviation from the mean.

    Anchoring on the mean is the conservative choice: the spatial loss is
    exactly the proven ``mean`` projector, and the residuals are zero-mean
    by construction (balanced positive/negative).
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 temporal_weight: float = 0.5, use_vla_norm: bool = False):
        super().__init__()
        self.temporal_weight = temporal_weight
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.temporal_pos_proj, self.residual_decoder = _make_temporal_modules(vae_dim)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = self.project_vla(vla_hidden)

        # ── spatial: cosine with temporal mean (same as mean) ─────
        vae_mean = vae_hidden.mean(dim=2)                              # [B, N, C]
        cos_sim = (F.normalize(vla_proj, dim=-1)
                   * F.normalize(vae_mean, dim=-1)).sum(-1)            # [B, N]
        spatial_loss = _masked_cosine_loss(cos_sim, mask)

        if T <= 1:
            return spatial_loss

        # ── temporal: L2 on residuals from mean ───────────────────
        vae_residuals = vae_hidden - vae_mean.unsqueeze(2)             # [B, N, T, C]

        t_cue = self.temporal_pos_proj(
            _sinusoidal_temporal_embed(T, C, vla_proj.device),
        )
        pred_residuals = self.residual_decoder(
            vla_proj.unsqueeze(2) + t_cue,
        )                                                              # [B, N, T, C]

        target_scale = vae_residuals.detach().norm(dim=-1, keepdim=True).mean(dim=2, keepdim=True).clamp(min=1e-6)
        normed_error = ((pred_residuals - vae_residuals) / target_scale).pow(2).mean(dim=(2, 3))

        if mask is not None:
            m = mask.float()
            temporal_loss = (normed_error * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            temporal_loss = normed_error.mean(-1)

        return spatial_loss + self.temporal_weight * temporal_loss.mean()


# ────────────────────────────────────────────────────────────────────────────
# 15-v2. Residual Temporal — current-frame anchor  (4n+1 aware)
# ────────────────────────────────────────────────────────────────────────────

class AnchorCurrentTemporalProjector(nn.Module):
    """Current-frame cosine + future-delta L2, designed for 4n+1 VAE.

    The WAN VAE compresses with a 4n+1 pattern: VAE[0] is the full-quality
    single-frame feature, while VAE[1..] are each an average of 4 frames.
    These are **heterogeneous** — VAE[0] is structurally different from the
    rest.  Using ``mean(VAE)`` as anchor blurs this distinction.

    This projector uses **VAE[0] as the anchor** instead:

    - **Spatial loss**: ``cosine(VLA_proj, VAE[0])`` — aligns with the
      precise current-frame observation (highest fidelity target).
    - **Temporal loss**: ``L2(decoder(VLA_proj + pos(t)), VAE[t] − VAE[0])``
      for ``t > 0`` — predicts *how the future differs from current*.

    The future deltas ``VAE[t] − VAE[0]`` have a clear semantic meaning:
    "what will change relative to now", which is directly action-relevant.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 temporal_weight: float = 0.5, use_vla_norm: bool = False):
        super().__init__()
        self.temporal_weight = temporal_weight
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.temporal_pos_proj, self.delta_decoder = _make_temporal_modules(vae_dim)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = self.project_vla(vla_hidden)

        # ── spatial: cosine with current frame (VAE[0], full quality) ─
        vae_current = vae_hidden[:, :, 0]                              # [B, N, C]
        cos_sim = (F.normalize(vla_proj, dim=-1)
                   * F.normalize(vae_current, dim=-1)).sum(-1)         # [B, N]
        spatial_loss = _masked_cosine_loss(cos_sim, mask)

        if T <= 1:
            return spatial_loss

        # ── temporal: L2 on future deltas relative to current ─────
        future_deltas = vae_hidden[:, :, 1:] - vae_current.unsqueeze(2)  # [B, N, T-1, C]
        T_fut = T - 1

        t_cue = self.temporal_pos_proj(
            _sinusoidal_temporal_embed(T_fut, C, vla_proj.device),
        )
        pred_deltas = self.delta_decoder(
            vla_proj.unsqueeze(2) + t_cue,
        )                                                              # [B, N, T-1, C]

        target_scale = future_deltas.detach().norm(dim=-1, keepdim=True).mean(dim=2, keepdim=True).clamp(min=1e-6)
        normed_error = ((pred_deltas - future_deltas) / target_scale).pow(2).mean(dim=(2, 3))

        if mask is not None:
            m = mask.float()
            temporal_loss = (normed_error * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            temporal_loss = normed_error.mean(-1)

        return spatial_loss + self.temporal_weight * temporal_loss.mean()

    @torch.no_grad()
    def reconstruct_frames(self, vla_hidden: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Inference helper: reconstruct per-frame VAE features.

        Returns [B, N, T', C] where frame 0 = vla_proj (current),
        frames 1.. = vla_proj + predicted delta.
        """
        vla_proj = self.project_vla(vla_hidden)
        C = vla_proj.shape[-1]
        if num_frames <= 1:
            return vla_proj.unsqueeze(2)
        t_cue = self.temporal_pos_proj(
            _sinusoidal_temporal_embed(num_frames - 1, C, vla_proj.device),
        )
        pred_deltas = self.delta_decoder(vla_proj.unsqueeze(2) + t_cue)
        future = vla_proj.unsqueeze(2) + pred_deltas
        return torch.cat([vla_proj.unsqueeze(2), future], dim=2)


# ────────────────────────────────────────────────────────────────────────────
# 15-v2. Anchor-Current with L2 spatial (exact reconstruction variant)
# ────────────────────────────────────────────────────────────────────────────

class AnchorCurrentTemporalL2Projector(nn.Module):
    """Variant with L2 spatial loss for exact per-frame reconstruction.

    Same as ``AnchorCurrentTemporalProjector`` but uses **L2** instead
    of cosine for the spatial anchor, so ``vla_proj`` matches VAE[0]
    in both direction and magnitude.  Enables precise reconstruction::

        frame[0] = vla_proj                        ≈ VAE[0]
        frame[t] = vla_proj + delta_decoder(t)     ≈ VAE[t]
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 temporal_weight: float = 0.5, use_vla_norm: bool = False):
        super().__init__()
        self.temporal_weight = temporal_weight
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.temporal_pos_proj, self.delta_decoder = _make_temporal_modules(vae_dim)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = self.project_vla(vla_hidden)

        # ── spatial: L2 with current frame ────────────────────────
        vae_current = vae_hidden[:, :, 0]
        spatial_error = (vla_proj - vae_current).pow(2).mean(dim=-1)
        if mask is not None:
            m = mask.float()
            spatial_loss = (spatial_error * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            spatial_loss = spatial_error.mean(-1)
        spatial_loss = spatial_loss.mean()

        if T <= 1:
            return spatial_loss

        # ── temporal: L2 on future deltas ─────────────────────────
        future_deltas = vae_hidden[:, :, 1:] - vae_current.unsqueeze(2)
        T_fut = T - 1

        t_cue = self.temporal_pos_proj(
            _sinusoidal_temporal_embed(T_fut, C, vla_proj.device),
        )
        pred_deltas = self.delta_decoder(
            vla_proj.unsqueeze(2) + t_cue,
        )

        target_scale = future_deltas.detach().norm(dim=-1, keepdim=True).mean(dim=2, keepdim=True).clamp(min=1e-6)
        normed_error = ((pred_deltas - future_deltas) / target_scale).pow(2).mean(dim=(2, 3))

        if mask is not None:
            temporal_loss = (normed_error * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            temporal_loss = normed_error.mean(-1)

        return spatial_loss + self.temporal_weight * temporal_loss.mean()

    @torch.no_grad()
    def reconstruct_frames(self, vla_hidden: torch.Tensor, num_frames: int) -> torch.Tensor:
        vla_proj = self.project_vla(vla_hidden)
        C = vla_proj.shape[-1]
        if num_frames <= 1:
            return vla_proj.unsqueeze(2)
        t_cue = self.temporal_pos_proj(
            _sinusoidal_temporal_embed(num_frames - 1, C, vla_proj.device),
        )
        pred_deltas = self.delta_decoder(vla_proj.unsqueeze(2) + t_cue)
        future = vla_proj.unsqueeze(2) + pred_deltas
        return torch.cat([vla_proj.unsqueeze(2), future], dim=2)


# ────────────────────────────────────────────────────────────────────────────
# 15-B. Full-Frame Temporal (mean cosine + per-frame L2 direct prediction)
# ────────────────────────────────────────────────────────────────────────────

class FullFrameTemporalProjector(nn.Module):
    """Variant B: cosine spatial + direct per-frame L2 prediction.

    Instead of predicting residuals, the temporal decoder directly predicts
    each frame's **absolute** VAE features::

        predicted[t] = frame_decoder(vla_proj + pos(t))    ≈ VAE[t]

    The primary cosine loss anchors the VLA projection to the spatial mean
    (proven stable).  The L2 per-frame loss trains the decoder to produce
    full frame features, inherently learning both spatial content and
    temporal variation in one branch.

    At inference::

        reconstructed[t] = frame_decoder(vla_proj + pos(t))

    No separate mean + residual composition needed — the decoder outputs
    complete frame features directly.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 temporal_weight: float = 0.5, use_vla_norm: bool = False):
        super().__init__()
        self.temporal_weight = temporal_weight
        self.fc1, self.act, self.fc2, self.vla_norm = _make_vla_proj_layers(vla_dim, vae_dim, use_vla_norm)
        self.temporal_pos_proj, self.frame_decoder = _make_temporal_modules(vae_dim)
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        vla_proj = self.project_vla(vla_hidden)

        # ── primary: cosine with temporal mean (same as mean) ─────
        vae_mean = vae_hidden.mean(dim=2)
        cos_sim = (F.normalize(vla_proj, dim=-1)
                   * F.normalize(vae_mean, dim=-1)).sum(-1)
        spatial_loss = _masked_cosine_loss(cos_sim, mask)

        if T <= 1:
            return spatial_loss

        # ── auxiliary: direct per-frame L2 prediction ─────────────
        t_cue = self.temporal_pos_proj(
            _sinusoidal_temporal_embed(T, C, vla_proj.device),
        )
        predicted = self.frame_decoder(
            vla_proj.unsqueeze(2) + t_cue,
        )                                                              # [B, N, T, C]

        target_scale = vae_hidden.detach().norm(dim=-1, keepdim=True).mean(dim=2, keepdim=True).clamp(min=1e-6)
        normed_error = ((predicted - vae_hidden) / target_scale).pow(2).mean(dim=(2, 3))

        if mask is not None:
            m = mask.float()
            temporal_loss = (normed_error * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            temporal_loss = normed_error.mean(-1)

        return spatial_loss + self.temporal_weight * temporal_loss.mean()

    @torch.no_grad()
    def reconstruct_frames(self, vla_hidden: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Inference helper: reconstruct per-frame VAE features.

        Args:
            vla_hidden: [B, N, D_vla] VLA hidden states.
            num_frames: T' — number of temporal frames to reconstruct.

        Returns:
            [B, N, T', C] predicted per-frame VAE features.
        """
        vla_proj = self.project_vla(vla_hidden)
        C = vla_proj.shape[-1]
        t_cue = self.temporal_pos_proj(
            _sinusoidal_temporal_embed(num_frames, C, vla_proj.device),
        )
        return self.frame_decoder(vla_proj.unsqueeze(2) + t_cue)


# ────────────────────────────────────────────────────────────────────────────
# Multi-Frame Conv  (iREPA-style: conv replaces MLP + spatial norm)
# ────────────────────────────────────────────────────────────────────────────

class ConvMultiFrameAlignProjector(nn.Module):
    """Conv-based multi-frame alignment following iREPA.

    Two key changes from ``multi_frame_concat``:

    1. **Conv replaces MLP**: a 1×1 channel projection + GELU + 3×3 spatial
       conv.  The 3×3 conv lets each spatial token see its 8 neighbours during
       projection, transferring spatial structure more faithfully than per-token
       MLP (which treats every patch independently).

    2. **Spatial normalization on VAE targets**: per-channel mean/std
       normalisation across the spatial (token) dimension with a tuneable
       ``gamma`` that controls how aggressively the spatial mean is removed::

           x = x - gamma * x.mean(dim=spatial)
           x = x / (x.std(dim=spatial) + eps)

       This accentuates *relative* spatial patterns over global magnitude.

    VLA tokens are reshaped to a 2-D spatial grid (H = W = √N) for convolution.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 3, num_cameras: int = 1,
                 spatial_norm_gamma: float = 1.0,
                 use_vla_norm: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        self.vae_dim = vae_dim
        self.spatial_norm_gamma = spatial_norm_gamma
        out_dim = num_frames * vae_dim

        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        self.conv1 = nn.Conv2d(num_cameras * vla_dim, out_dim, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_dim, num_cameras * out_dim, kernel_size=3, padding=1)

        _init_weights(self)

    @staticmethod
    def _spatial_norm(x: torch.Tensor, gamma: float) -> torch.Tensor:
        """iREPA spatial normalization: normalise across token dim (dim=1).

        x: [B, N, D]  →  zero-centred, unit-variance along N for each (B, D).
        """
        x = x - gamma * x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)
        return x

    def _camera_grid(self, N):
        G = self.num_cameras
        side = int((N // G) ** 0.5)
        return G, side, side

    def project_vla(self, h):
        if self.vla_norm is not None:
            h = self.vla_norm(h)
        B, N, D = h.shape
        G, H, W = self._camera_grid(N)
        x = h.reshape(B, G, H, W, D).permute(0, 4, 1, 2, 3).reshape(B, G * D, H, W)
        x = self.conv2(self.act(self.conv1(x)))
        x = x.reshape(B, G, self.num_frames, self.vae_dim, H, W)
        x = x.permute(0, 1, 4, 5, 2, 3).reshape(B, N, self.num_frames, self.vae_dim)
        return x.mean(dim=2)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        G, H, W = self._camera_grid(N)

        h = vla_hidden
        if self.vla_norm is not None:
            h = self.vla_norm(h)

        D = h.shape[-1]
        x = h.reshape(B, G, H, W, D).permute(0, 4, 1, 2, 3).reshape(B, G * D, H, W)
        x = self.conv2(self.act(self.conv1(x)))               # [B, G*nf*C, H, W]

        x = x.reshape(B, G, self.num_frames, C, H, W)
        predicted = x.permute(0, 1, 4, 5, 2, 3).reshape(B, N, self.num_frames, C)

        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)

        if self.spatial_norm_gamma > 0:
            vae_for_loss = torch.stack(
                [self._spatial_norm(vae_hidden[:, :, t, :], self.spatial_norm_gamma)
                 for t in range(T)], dim=2,
            )
        else:
            vae_for_loss = vae_hidden

        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_for_loss, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# Multi-Frame Conv + Spatial PE + Temporal Refine
# ────────────────────────────────────────────────────────────────────────────

class ConvSpatialPETemporalRefineProjector(nn.Module):
    """Conv projection with spatial PE on input and temporal PE + refinement on output.

    Builds on ``ConvMultiFrameAlignProjector`` with two additions:

    1. **Spatial PE on input**: 2-D sinusoidal UV positional embedding is added
       to VLA features *before* the conv layers.  The 3×3 conv only has a
       local receptive field; spatial PE gives it global position awareness
       so the projection can be position-dependent (e.g. predicting different
       features for "top-left table" vs "bottom-right gripper").

    2. **Temporal PE on output**: after the conv produces per-frame features,
       sinusoidal temporal embeddings are added and passed through a lightweight
       refinement MLP.  This gives each frame an explicit temporal identity
       (frame 0 = current, frame 1 = near future, frame 2 = further future)
       instead of relying on implicit channel ordering.

    No information bottleneck — the conv directly outputs T'×C channels.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 3, num_cameras: int = 1,
                 spatial_norm_gamma: float = 1.0,
                 spatial_pe_ratio: float = 0.1,
                 use_vla_norm: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        self.vae_dim = vae_dim
        self.spatial_norm_gamma = spatial_norm_gamma
        self.spatial_pe_ratio = spatial_pe_ratio
        out_dim = num_frames * vae_dim

        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        self.conv1 = nn.Conv2d(num_cameras * vla_dim, out_dim, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_dim, num_cameras * out_dim, kernel_size=3, padding=1)

        self.temporal_pos_proj = nn.Sequential(
            nn.Linear(vae_dim, vae_dim), nn.SiLU(), nn.Linear(vae_dim, vae_dim),
        )
        self.frame_refine = nn.Sequential(
            nn.Linear(vae_dim, vae_dim * 2), nn.GELU(), nn.Linear(vae_dim * 2, vae_dim),
        )

        _init_weights(self)

    @staticmethod
    def _spatial_norm(x: torch.Tensor, gamma: float) -> torch.Tensor:
        x = x - gamma * x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)
        return x

    def _camera_grid(self, N):
        G = self.num_cameras
        side = int((N // G) ** 0.5)
        return G, side, side

    def _apply_spatial_pe_per_camera(self, h):
        """Apply spatial PE per camera, then stack channels: [B, G*D, H, W]."""
        B, N, D = h.shape
        G, H, W = self._camera_grid(N)
        x = h.reshape(B * G, H * W, D).permute(0, 2, 1).reshape(B * G, D, H, W)
        x = apply_spatial_pe(x, ratio=self.spatial_pe_ratio)
        return x.reshape(B, G * D, H, W)

    def project_vla(self, h):
        if self.vla_norm is not None:
            h = self.vla_norm(h)
        B, N, D = h.shape
        G, H, W = self._camera_grid(N)
        x = self._apply_spatial_pe_per_camera(h)
        x = self.conv2(self.act(self.conv1(x)))                    # [B, G*nf*C, H, W]
        x = x.reshape(B, G, self.num_frames, self.vae_dim, H, W)
        x = x.permute(0, 1, 4, 5, 2, 3).reshape(B, N, self.num_frames, self.vae_dim)
        return x.mean(dim=2)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        G, H, W = self._camera_grid(N)

        h = vla_hidden
        if self.vla_norm is not None:
            h = self.vla_norm(h)

        x = self._apply_spatial_pe_per_camera(h)
        x = self.conv2(self.act(self.conv1(x)))                    # [B, G*nf*C, H, W]

        x = x.reshape(B, G, self.num_frames, C, H, W)
        predicted = x.permute(0, 1, 4, 5, 2, 3).reshape(B, N, self.num_frames, C)

        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)

        # temporal PE → refinement
        t_pe = self.temporal_pos_proj(
            _sinusoidal_temporal_embed(T, C, predicted.device))    # [T, C]
        predicted = predicted + t_pe
        predicted = predicted + self.frame_refine(predicted)

        if self.spatial_norm_gamma > 0:
            vae_for_loss = torch.stack(
                [self._spatial_norm(vae_hidden[:, :, t, :], self.spatial_norm_gamma)
                 for t in range(T)], dim=2,
            )
        else:
            vae_for_loss = vae_hidden

        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_for_loss, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)


# ────────────────────────────────────────────────────────────────────────────
# All-Conv: Spatial PE + Direct Output + Temporal Conv Mixing
# ────────────────────────────────────────────────────────────────────────────

class ConvSpatialPETemporalConvProjector(nn.Module):
    """All-conv projector with spatial PE and temporal conv mixing.

    Pipeline (all convolutional, no MLP, no bottleneck):

    1. **Spatial PE + Conv2d direct output**: 2-D sinusoidal spatial PE is added
       to VLA features.  Conv2d 1×1 → GELU → Conv2d 3×3 directly outputs T'×C
       channels per spatial location.  Each frame has its own dedicated output
       channels — no shared representation.

    2. **Temporal Conv1d mixing**: after reshaping to per-frame features
       ``[B*N, C, T]``, a 1-D convolution across the temporal axis lets
       adjacent frames refine each other.  With T=3 and kernel_size=3, every
       frame sees all other frames in a single pass.  Applied as a residual.

    This combines iREPA's spatial-structure insight (conv + spatial PE) with
    explicit temporal coherence (temporal conv), while keeping the direct
    multi-frame output path that experimentally outperforms bottleneck designs.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 3, num_cameras: int = 1,
                 spatial_norm_gamma: float = 1.0,
                 spatial_pe_ratio: float = 0.1,
                 use_vla_norm: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        self.vae_dim = vae_dim
        self.spatial_norm_gamma = spatial_norm_gamma
        self.spatial_pe_ratio = spatial_pe_ratio
        out_dim = num_frames * vae_dim

        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        self.conv1 = nn.Conv2d(num_cameras * vla_dim, out_dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_dim * 4, num_cameras * out_dim, kernel_size=3, padding=1)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(vae_dim, vae_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(vae_dim, vae_dim, kernel_size=1),
        )

        _init_weights(self)

    @staticmethod
    def _spatial_norm(x: torch.Tensor, gamma: float) -> torch.Tensor:
        x = x - gamma * x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)
        return x

    def _camera_grid(self, N):
        G = self.num_cameras
        side = int((N // G) ** 0.5)
        return G, side, side

    def _apply_spatial_pe_per_camera(self, h):
        B, N, D = h.shape
        G, H, W = self._camera_grid(N)
        x = h.reshape(B * G, H * W, D).permute(0, 2, 1).reshape(B * G, D, H, W)
        x = apply_spatial_pe(x, ratio=self.spatial_pe_ratio)
        return x.reshape(B, G * D, H, W)

    def project_vla(self, h):
        if self.vla_norm is not None:
            h = self.vla_norm(h)
        B, N, D = h.shape
        G, H, W = self._camera_grid(N)
        x = self._apply_spatial_pe_per_camera(h)
        x = self.conv2(self.act(self.conv1(x)))
        x = x.reshape(B, G, self.num_frames, self.vae_dim, H, W)
        x = x.permute(0, 1, 4, 5, 2, 3).reshape(B, N, self.num_frames, self.vae_dim)
        return x.mean(dim=2)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        G, H, W = self._camera_grid(N)

        h = vla_hidden
        if self.vla_norm is not None:
            h = self.vla_norm(h)

        x = self._apply_spatial_pe_per_camera(h)
        x = self.conv2(self.act(self.conv1(x)))                    # [B, G*T*C, H, W]

        x = x.reshape(B, G, self.num_frames, C, H, W)
        predicted = x.permute(0, 1, 4, 5, 2, 3).reshape(B, N, self.num_frames, C)

        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)

        # temporal conv mixing (residual): [B*N, C, T]
        p_flat = predicted.reshape(B * N, T, C).permute(0, 2, 1)
        predicted = predicted + self.temporal_conv(p_flat).permute(0, 2, 1).reshape(B, N, T, C)

        if self.spatial_norm_gamma > 0:
            vae_for_loss = torch.stack(
                [self._spatial_norm(vae_hidden[:, :, t, :], self.spatial_norm_gamma)
                 for t in range(T)], dim=2,
            )
        else:
            vae_for_loss = vae_hidden

        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_for_loss, dim=-1)).sum(-1).mean(-1)
        return _masked_cosine_loss(cos_sim, mask)
class MultiFrameConcatResidualDynamicsProjector(nn.Module):
    """Improved multi-frame concat with three enhancements over the original:

    1. **Residual decomposition**: a shared ``mean_head`` predicts the temporal
       mean, and a separate ``delta_head`` predicts per-frame residuals.
       ``predicted[t] = mean_pred + delta_pred[t]``.  This decomposes
       "where" (easy, shared) from "how it changes" (hard, per-frame),
       making the delta MLP's job much simpler.

    2. **Temporal mixer**: a lightweight 1-D convolution across the T
       dimension after the initial per-frame prediction, allowing
       cross-frame refinement (smoothness / coherence) that the flat
       concat cannot express.

    3. **Dynamics auxiliary loss**: an extra linear head predicts the
       normalised temporal change direction
       ``target = norm(mean(VAE[1:]) − VAE[0])``, encouraging the VLA
       token to explicitly encode motion direction.

    No information bottleneck — all heads receive the full VLA hidden.
    """

    multi_frame = True

    def __init__(self, vla_dim: int, vae_dim: int = 48,
                 num_frames: int = 3, dynamics_weight: float = 0.3,
                 use_vla_norm: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.vae_dim = vae_dim
        self.dynamics_weight = dynamics_weight

        hidden = vae_dim * 4
        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None

        self.mean_fc1 = nn.Linear(vla_dim, hidden)
        self.mean_act = nn.GELU()
        self.mean_fc2 = nn.Linear(hidden, vae_dim)

        delta_out = num_frames * vae_dim
        self.delta_fc1 = nn.Linear(vla_dim, delta_out * 4)
        self.delta_act = nn.GELU()
        self.delta_fc2 = nn.Linear(delta_out * 4, delta_out)

        self.temporal_mixer = nn.Sequential(
            nn.Conv1d(vae_dim, vae_dim, kernel_size=3, padding=1, groups=vae_dim),
            nn.GELU(),
            nn.Conv1d(vae_dim, vae_dim, kernel_size=1),
        )

        self.dynamics_head = nn.Linear(vae_dim, vae_dim)

        _init_weights(self)

    def project_vla(self, h):
        if self.vla_norm is not None:
            h = self.vla_norm(h)
        return self.mean_fc2(self.mean_act(self.mean_fc1(h)))

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        h = self.vla_norm(vla_hidden) if self.vla_norm is not None else vla_hidden

        mean_pred = self.mean_fc2(self.mean_act(self.mean_fc1(h)))
        delta_pred = self.delta_fc2(self.delta_act(self.delta_fc1(h)))
        delta_pred = delta_pred.reshape(B, N, self.num_frames, C)

        predicted = mean_pred.unsqueeze(2) + delta_pred

        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)

        # temporal mixer: [B*N, C, T] conv → [B*N, C, T]
        p_flat = predicted.reshape(B * N, T, C).permute(0, 2, 1)
        p_mixed = self.temporal_mixer(p_flat).permute(0, 2, 1).reshape(B, N, T, C)
        predicted = predicted + p_mixed

        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)).sum(-1).mean(-1)
        main_loss = _masked_cosine_loss(cos_sim, mask)

        if T <= 1 or self.dynamics_weight == 0:
            return main_loss

        temporal_dir = F.normalize(
            vae_hidden[:, :, 1:].mean(2) - vae_hidden[:, :, 0], dim=-1,
        )
        pred_dir = F.normalize(self.dynamics_head(mean_pred), dim=-1)
        dir_cos = (pred_dir * temporal_dir).sum(-1)
        dynamics_loss = _masked_cosine_loss(dir_cos, mask)

        return main_loss + self.dynamics_weight * dynamics_loss


# ────────────────────────────────────────────────────────────────────────────
# Static-Dynamic Decomposition with Temporal Difference Alignment
# ────────────────────────────────────────────────────────────────────────────

class StaticDynamicDecompProjector(nn.Module):
    """Multi-frame prediction decomposed into static base + per-frame delta.

    **Core idea**: VAE video latents have a natural static/dynamic structure.
    Most channels encode slowly-changing content (background, idle objects),
    while a few encode fast-changing content (gripper, manipulated object).
    A flat MLP wastes capacity treating all T'×C outputs equally.

    This projector decomposes the prediction into two branches::

        s       = MLP_static(vla_hidden)       # [B, N, C]     — static base
        Δ_t     = MLP_delta(vla_hidden)[t]     # [B, N, T', C] — per-frame residual
        pred[t] = s + Δ_t                      # [B, N, T', C]

    The static branch is a small MLP (vla_dim → C) that predicts what is
    shared across frames.  The delta branch (vla_dim → T'×C) only needs to
    predict **deviations** from that base — a much simpler task for the
    dynamic channels, and near-zero output for static channels.

    Loss = ``L_frame + diff_weight · L_diff``

    - ``L_frame``:  per-frame cosine alignment (same as ``multi_frame_concat``).
    - ``L_diff``:   cosine alignment of *inter-frame differences*
      (``pred[t+1]−pred[t]`` vs ``vae[t+1]−vae[t]``).

    ``L_diff`` is the key addition.  Standard per-frame cosine loss is
    **order-invariant** — shuffling the T' frames gives the same loss.
    ``L_diff`` explicitly supervises temporal dynamics and naturally focuses
    on dynamic channels: the cosine direction of ``vae[t+1]−vae[t]`` is
    dominated by the few channels with large temporal change.

    **Key differences from ``concat_residual_dynamics``**:

    1. *Temporal difference loss* (per-step ``cos(Δ_pred, Δ_vae)``), not a
       single global direction vector.
    2. *``project_vla`` returns full composition* ``s + Δ`` for inference;
       ``concat_residual_dynamics.project_vla`` only returns the static mean.
    3. No temporal mixer — keeps the module lightweight and lets the loss
       structure (not architecture complexity) drive temporal quality.
    """

    multi_frame = True

    def __init__(
        self,
        vla_dim: int,
        vae_dim: int = 48,
        num_frames: int = 2,
        use_vla_norm: bool = False,
        diff_weight: float = 0.5,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.vae_dim = vae_dim
        self.diff_weight = diff_weight

        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None

        hidden_s = vae_dim * 4
        self.static_fc1 = nn.Linear(vla_dim, hidden_s)
        self.static_act = nn.GELU()
        self.static_fc2 = nn.Linear(hidden_s, vae_dim)

        delta_out = num_frames * vae_dim
        self.delta_fc1 = nn.Linear(vla_dim, delta_out * 4)
        self.delta_act = nn.GELU()
        self.delta_fc2 = nn.Linear(delta_out * 4, delta_out)

        _init_weights(self)

    def _compose(self, h: torch.Tensor) -> torch.Tensor:
        """Compose per-frame predictions.  Returns ``[B, N, T', C]``."""
        if self.vla_norm is not None:
            h = self.vla_norm(h)
        s = self.static_fc2(self.static_act(self.static_fc1(h)))
        delta = self.delta_fc2(self.delta_act(self.delta_fc1(h)))
        delta = delta.reshape(*delta.shape[:-1], self.num_frames, self.vae_dim)
        return s.unsqueeze(-2) + delta

    def project_vla(self, h: torch.Tensor) -> torch.Tensor:
        return self._compose(h).reshape(*h.shape[:-1], -1)

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        predicted = self._compose(vla_hidden)

        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)

        # L_frame: per-frame cosine alignment
        cos_frame = (
            F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)
        ).sum(-1).mean(-1)
        frame_loss = _masked_cosine_loss(cos_frame, mask)

        # L_diff: temporal difference alignment
        if T > 1:
            pred_diff = predicted[:, :, 1:] - predicted[:, :, :-1]
            vae_diff = vae_hidden[:, :, 1:] - vae_hidden[:, :, :-1]
            cos_diff = (
                F.normalize(pred_diff, dim=-1) * F.normalize(vae_diff, dim=-1)
            ).sum(-1).mean(-1)
            diff_loss = _masked_cosine_loss(cos_diff, mask)
        else:
            diff_loss = frame_loss.new_tensor(0.0)

        return frame_loss + self.diff_weight * diff_loss


# ────────────────────────────────────────────────────────────────────────────
# Temporal Contrastive MultiFrameConcat  (per-frame T'×T' InfoNCE)
# ────────────────────────────────────────────────────────────────────────────

class TemporalContrastiveConcatProjector(nn.Module):
    """MultiFrameConcat + per-frame temporal InfoNCE auxiliary loss.

    Base MLP is identical to ``MultiFrameConcatAlignProjector``.

    The auxiliary loss builds a **T' × T' similarity matrix** at each
    spatial token::

        S[t, t'] = cos(predicted[t], vae[t']) / τ

    and applies per-row cross-entropy with the diagonal as the correct
    label.  Each ``predicted[t]`` must be most similar to ``vae[t]``
    (its own timestep) rather than ``vae[t']`` at other timesteps.

    **Complementary to L_frame**: L_frame pulls ``predicted[t]`` toward
    ``vae[t]`` (attraction only).  The contrastive loss additionally
    pushes ``predicted[t]`` *away* from ``vae[t']`` for ``t' ≠ t``
    (attraction + repulsion).

    **Self-adaptive**: for static spatial tokens where
    ``vae[0] ≈ vae[1] ≈ vae[2]``, the similarity matrix is uniform,
    softmax outputs a flat distribution, and the gradient is ≈ 0.
    The loss only produces signal where real temporal variation exists.

    Loss = ``L_frame + contrast_weight · L_contrast``
    """

    multi_frame = True

    def __init__(
        self,
        vla_dim: int,
        vae_dim: int = 48,
        num_frames: int = 2,
        use_vla_norm: bool = False,
        contrast_weight: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.vae_dim = vae_dim
        self.contrast_weight = contrast_weight
        self.temperature = temperature

        out_dim = num_frames * vae_dim
        self.fc1 = nn.Linear(vla_dim, out_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim * 4, out_dim)
        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def _temporal_contrastive_loss(
        self,
        predicted: torch.Tensor,
        vae_hidden: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-frame T' × T' temporal InfoNCE.

        For each spatial token, builds a [T', T'] similarity matrix between
        predicted frames (queries) and VAE frames (keys).  Each row is a
        T'-way classification: predicted[t] should match vae[t].
        """
        B, N, T, C = predicted.shape
        if T <= 1:
            return predicted.new_tensor(0.0)

        BN = B * N
        q = F.normalize(predicted.reshape(BN, T, C), dim=-1)
        k = F.normalize(vae_hidden.reshape(BN, T, C), dim=-1)

        logits = torch.bmm(q, k.transpose(1, 2)) / self.temperature   # [BN, T, T]
        labels = torch.arange(T, device=logits.device).expand(BN, -1) # [BN, T]

        ce = F.cross_entropy(
            logits.reshape(BN * T, T), labels.reshape(BN * T),
            reduction="none",
        ).reshape(B, N, T).mean(-1)                                    # [B, N]

        if mask is not None:
            m = mask.float()
            loss = (ce * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            loss = ce.mean(-1)
        return loss.mean()

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        predicted = self.project_vla(vla_hidden).reshape(B, N, self.num_frames, C)
        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)

        cos_sim = (
            F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)
        ).sum(-1).mean(-1)
        frame_loss = _masked_cosine_loss(cos_sim, mask)

        contrast_loss = self._temporal_contrastive_loss(predicted, vae_hidden, mask)

        self._last_frame_loss = frame_loss.detach()
        self._last_contrast_loss = contrast_loss.detach()

        # return frame_loss + self.contrast_weight * contrast_loss
        return self.contrast_weight * contrast_loss


# ────────────────────────────────────────────────────────────────────────────
# Temporal Contrastive Soft-Label variant
# ────────────────────────────────────────────────────────────────────────────

class SoftTemporalContrastiveConcatProjector(nn.Module):
    """Same as TemporalContrastiveConcatProjector but with soft temporal labels.

    Hard labels treat all non-diagonal frames as equally wrong.  With T'=3
    this means vae[0] and vae[2] are penalised identically for predicted[1],
    even though vae[0] is temporally adjacent and much more similar.

    Soft labels use ``softmax(−|t−t'| / σ)`` so nearby frames get partial
    credit.  For T'=3, σ=0.5 the target for predicted[1] is roughly
    ``[0.21, 0.58, 0.21]`` instead of ``[0, 1, 0]``.
    """

    multi_frame = True

    def __init__(
        self,
        vla_dim: int,
        vae_dim: int = 48,
        num_frames: int = 2,
        use_vla_norm: bool = False,
        contrast_weight: float = 0.1,
        temperature: float = 0.07,
        soft_label_sigma: float = 0.5,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.vae_dim = vae_dim
        self.contrast_weight = contrast_weight
        self.temperature = temperature
        self.soft_label_sigma = soft_label_sigma

        out_dim = num_frames * vae_dim
        self.fc1 = nn.Linear(vla_dim, out_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim * 4, out_dim)
        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    def _temporal_contrastive_loss(self, predicted, vae_hidden, mask):
        B, N, T, C = predicted.shape
        if T <= 1:
            return predicted.new_tensor(0.0)

        BN = B * N
        q = F.normalize(predicted.reshape(BN, T, C), dim=-1)
        k = F.normalize(vae_hidden.reshape(BN, T, C), dim=-1)

        logits = torch.bmm(q, k.transpose(1, 2)) / self.temperature  # [BN, T, T]

        idx = torch.arange(T, device=logits.device, dtype=torch.float32)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        targets = F.softmax(-dist / self.soft_label_sigma, dim=-1)    # [T, T]
        log_probs = F.log_softmax(logits, dim=-1)                     # [BN, T, T]
        ce = -(targets * log_probs).sum(-1).reshape(B, N, T).mean(-1) # [B, N]

        if mask is not None:
            m = mask.float()
            loss = (ce * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            loss = ce.mean(-1)
        return loss.mean()

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        predicted = self.project_vla(vla_hidden).reshape(B, N, self.num_frames, C)
        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)

        cos_sim = (
            F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)
        ).sum(-1).mean(-1)
        frame_loss = _masked_cosine_loss(cos_sim, mask)

        contrast_loss = self._temporal_contrastive_loss(predicted, vae_hidden, mask)

        self._last_frame_loss = frame_loss.detach()
        self._last_contrast_loss = contrast_loss.detach()

        scale = frame_loss.detach() / contrast_loss.detach().clamp(min=1e-6)
        return frame_loss + self.contrast_weight * scale * contrast_loss


# ────────────────────────────────────────────────────────────────────────────
# Time-Contrastive Learning (TCL) enhanced multi-frame concat projector
# ────────────────────────────────────────────────────────────────────────────

class TCLMultiFrameConcatProjector(nn.Module):
    """MultiFrameConcat enhanced with Time-Contrastive Learning (TCL).

    Combines per-frame cosine alignment (from ``multi_frame_concat``) with a
    time-contrastive objective (Sermanet et al., 2018; Nair et al., 2022;
    Ma et al., 2023) that encourages each per-frame "moment token" to encode
    temporally discriminative cues.

    For each timestep *t* within a trajectory:

    - **Anchor** ``z_t = g(predicted_t)``, the projection-head output of the
      VLA-predicted feature at timestep *t*.
    - **Positive** ``z⁺_t = g(aug(vae_t))``, from an augmented view of the
      same-timestep VAE feature.  Augmentations operate in feature-space and
      mirror photometric / noise / occlusion image-space perturbations.
    - **Hard negative** ``z⁻_t = g(vae_{t'})``, the most-similar feature at
      a *different* timestep *t' ≠ t* within the same trajectory.

    TCL loss (Eq. 4)::

        L_TCL = −Σ_t log[ exp(sim(z_t, z⁺_t)/τ)
                           / (exp(sim(z_t, z⁺_t)/τ) + exp(sim(z_t, z⁻_t)/τ)) ]

    Total loss = L_align + λ_tcl · L_TCL.

    The projection head ``g(·)`` and the augmentation branch are the only
    additional parameters.  The VLM backbone is assumed frozen during TCL
    pre-training; only the projector learns.
    """

    multi_frame = True

    def __init__(
        self,
        vla_dim: int,
        vae_dim: int = 48,
        num_frames: int = 2,
        use_vla_norm: bool = False,
        # TCL hyper-parameters
        temperature: float = 0.07,
        tcl_weight: float = 0.1,
        proj_dim: int = 128,
        hard_negative: bool = True,
        # feature-space augmentation
        aug_noise_std: float = 0.1,
        aug_dropout: float = 0.1,
        aug_scale_lo: float = 0.8,
        aug_scale_hi: float = 1.2,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.vae_dim = vae_dim
        self.temperature = temperature
        self.tcl_weight = tcl_weight
        self.hard_negative = hard_negative
        self.aug_noise_std = aug_noise_std
        self.aug_dropout = aug_dropout
        self.aug_scale_lo = aug_scale_lo
        self.aug_scale_hi = aug_scale_hi

        out_dim = num_frames * vae_dim
        self.fc1 = nn.Linear(vla_dim, out_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim * 4, out_dim)
        self.vla_norm = nn.LayerNorm(vla_dim) if use_vla_norm else None

        self.tcl_proj = nn.Sequential(
            nn.Linear(vae_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        _init_weights(self)

    def project_vla(self, h):
        return _project_vla(h, self.fc1, self.act, self.fc2, self.vla_norm)

    # ── feature-space augmentation (mirrors image-space perturbations) ──

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic feature-space augmentations to build positive pairs.

        Analogues: random scale → photometric; Gaussian noise → sensor noise;
        feature dropout → occlusion.
        """
        if not self.training:
            return x
        scale = x.new_empty(x.shape[0], 1, 1).uniform_(self.aug_scale_lo, self.aug_scale_hi)
        x = x * scale
        if self.aug_noise_std > 0:
            x = x + torch.randn_like(x) * self.aug_noise_std
        if self.aug_dropout > 0:
            keep = torch.bernoulli(x.new_full(x.shape, 1.0 - self.aug_dropout))
            x = x * keep / (1.0 - self.aug_dropout)
        return x

    # ── TCL loss (Eq. 4) ────────────────────────────────────────────────

    def _tcl_loss(
        self,
        predicted: torch.Tensor,
        vae_hidden: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Time-contrastive loss.

        Args:
            predicted:  [B, N, T', C]  per-frame predicted features.
            vae_hidden: [B, N, T', C]  per-frame VAE ground-truth features.
            mask:       [B, N] or None.
        """
        B, N, T, C = predicted.shape
        if T <= 1:
            return predicted.new_tensor(0.0)

        BN = B * N
        pred_flat = predicted.reshape(BN, T, C)
        vae_flat = vae_hidden.reshape(BN, T, C)

        z_anchor = F.normalize(self.tcl_proj(pred_flat), dim=-1)                 # [BN, T, D]
        z_pos = F.normalize(self.tcl_proj(self._augment(vae_flat)), dim=-1)      # [BN, T, D]
        sim_pos = (z_anchor * z_pos).sum(-1) / self.temperature                  # [BN, T]

        z_vae = F.normalize(self.tcl_proj(vae_flat), dim=-1)                     # [BN, T, D]

        if self.hard_negative:
            sim_all = torch.bmm(z_anchor, z_vae.transpose(1, 2)) / self.temperature  # [BN, T, T]
            eye_mask = torch.eye(T, device=predicted.device, dtype=torch.bool).unsqueeze(0)
            sim_neg = sim_all.masked_fill(eye_mask, float('-inf')).max(dim=-1).values  # [BN, T]
        else:
            shift = torch.randint(1, T, (BN,), device=predicted.device)
            neg_idx = (torch.arange(T, device=predicted.device).unsqueeze(0) + shift.unsqueeze(1)) % T
            z_neg = z_vae.gather(1, neg_idx.unsqueeze(-1).expand(-1, -1, z_anchor.shape[-1]))
            sim_neg = (z_anchor * z_neg).sum(-1) / self.temperature               # [BN, T]

        logits = torch.stack([sim_pos, sim_neg], dim=-1)                          # [BN, T, 2]
        loss_per_t = -sim_pos + torch.logsumexp(logits, dim=-1)                   # [BN, T]

        loss = loss_per_t.mean(dim=-1).reshape(B, N)
        if mask is not None:
            m = mask.float()
            loss = (loss * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            loss = loss.mean(-1)
        return loss.mean()

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, vla_hidden, vae_hidden, mask=None):
        B, N, T, C = vae_hidden.shape
        predicted = self.project_vla(vla_hidden).reshape(B, N, self.num_frames, C)
        if self.num_frames != T:
            predicted = F.interpolate(
                predicted.permute(0, 3, 1, 2), size=(N, T), mode="nearest",
            ).permute(0, 2, 3, 1)

        cos_sim = (F.normalize(predicted, dim=-1) * F.normalize(vae_hidden, dim=-1)).sum(-1).mean(-1)
        align_loss = _masked_cosine_loss(cos_sim, mask)

        tcl_loss = self._tcl_loss(predicted, vae_hidden, mask)

        return align_loss + self.tcl_weight * tcl_loss


# ────────────────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[nn.Module]] = {
    "mean": VideoAlignProjector,
    "max_cosine": MultiFrameCosineProjector,
    "softmax_cosine": MultiFrameCosineProjector,
    "temporal_reconstruct": TemporalReconstructAlignProjector,
    "multi_frame_concat": MultiFrameConcatAlignProjector,
    "multi_frame_concat_vla_anchored": MultiFrameConcatVlaAnchoredProjector,
    "multi_frame_spatial_pe": MultiFrameSpatialPEProjector,
    "multi_frame_temporal_pe": MultiFrameTemporalPEProjector,
    "multi_frame_vla_spatial_pe": MultiFrameVLASpatialPEProjector,
    "multi_frame_vla_temporal_pe_refine": MultiFrameVLATemporalPERefineProjector,
    "multi_frame_conv_direct": ConvDirectMultiFrameProjector,
    "concat_residual_dynamics": MultiFrameConcatResidualDynamicsProjector,
    "conv_multi_frame": ConvMultiFrameAlignProjector,
    "conv_spatial_pe_temporal_refine": ConvSpatialPETemporalRefineProjector,
    "conv_spatial_pe_temporal_conv": ConvSpatialPETemporalConvProjector,
    "temporal_flow": TemporalFlowAlignProjector,
    "autoregressive": AutoregressiveAlignProjector,
    "variational": VariationalTemporalAlignProjector,
    "contrastive": ContrastiveTemporalAlignProjector,
    "cross_attention": CrossAttentionReconstructProjector,
    "current_aware": CurrentFrameAwareAlignProjector,
    "causal": CausalPredictionAlignProjector,
    "dual_head": DualHeadAlignProjector,
    "temporal_decay": TemporalDecayCosineProjector,
    "mean_dynamics": MeanDynamicsProjector,
    "multi_scale_mean": MultiScaleTemporalMeanProjector,
    "residual_temporal": ResidualTemporalMeanProjector,
    "anchor_current": AnchorCurrentTemporalProjector,
    "residual_temporal_l2": AnchorCurrentTemporalL2Projector,
    "full_frame_temporal": FullFrameTemporalProjector,
    "dual_head_concat": DualHeadConcatAlignProjector,
    "tcl_multi_frame_concat": TCLMultiFrameConcatProjector,
    "gated_decomp": StaticDynamicDecompProjector,
    "temporal_contrastive_concat": TemporalContrastiveConcatProjector,
    "soft_temporal_contrastive_concat": SoftTemporalContrastiveConcatProjector,
}

_DELTA_AWARE_MODES = {"current_aware", "causal"}
_NUM_FRAMES_MODES = {"multi_frame_concat", "multi_frame_concat_vla_anchored", "multi_frame_spatial_pe", "multi_frame_temporal_pe", "multi_frame_vla_spatial_pe", "multi_frame_vla_temporal_pe_refine", "multi_frame_conv_direct", "concat_residual_dynamics", "conv_multi_frame", "conv_spatial_pe_temporal_refine", "conv_spatial_pe_temporal_conv", "dual_head_concat", "tcl_multi_frame_concat", "gated_decomp", "temporal_contrastive_concat", "soft_temporal_contrastive_concat"}
_CONV_MODES = {"multi_frame_conv_direct", "conv_multi_frame", "conv_spatial_pe_temporal_refine", "conv_spatial_pe_temporal_conv"}
_DUAL_HEAD_MODES = {"dual_head_concat"}


def create_video_align_projector(
    mode: str,
    vla_dim: int,
    vae_dim: int = 48,
    **kwargs,
) -> nn.Module:
    """Instantiate a video alignment projector by name.

    Args:
        mode: one of the keys in ``_REGISTRY``.
        vla_dim: VLA hidden dimension.
        vae_dim: VAE channel dimension (default 48).
        **kwargs: forwarded to the chosen projector constructor.
            ``video_delta_frames`` is automatically routed to delta-aware
            projectors and stripped for others.
    """
    if mode not in _REGISTRY:
        raise ValueError(f"Unknown video_align_mode {mode!r}. Choose from {sorted(_REGISTRY)}")

    delta_frames = kwargs.pop("video_delta_frames", None)
    delta_frames_aux = kwargs.pop("video_delta_frames_aux", None)
    if mode in _DELTA_AWARE_MODES:
        kwargs["video_delta_frames"] = delta_frames
    if mode in _NUM_FRAMES_MODES and delta_frames is not None:
        kwargs["num_frames"] = len(_compute_vae_representative_deltas(delta_frames))

    if mode in _DUAL_HEAD_MODES:
        if delta_frames_aux is not None:
            kwargs["num_frames_aux"] = len(_compute_vae_representative_deltas(delta_frames_aux))
        else:
            kwargs.setdefault("num_frames_aux", kwargs.get("num_frames", 2))
    else:
        kwargs.pop("num_frames_aux", None)
        kwargs.pop("loss_weight_primary", None)
        kwargs.pop("loss_weight_aux", None)

    if mode not in _CONV_MODES:
        kwargs.pop("num_cameras", None)

    _CONTRASTIVE_MODES = {"temporal_contrastive_concat", "soft_temporal_contrastive_concat"}
    if mode not in _CONTRASTIVE_MODES:
        kwargs.pop("contrast_weight", None)
        kwargs.pop("temperature", None)
        kwargs.pop("soft_label_sigma", None)
    if mode == "temporal_contrastive_concat":
        kwargs.pop("soft_label_sigma", None)

    _DIFF_MODES = {"gated_decomp"}
    if mode not in _DIFF_MODES:
        kwargs.pop("diff_weight", None)

    cls = _REGISTRY[mode]
    if mode == "max_cosine":
        kwargs.setdefault("temporal_pool", "max")
    elif mode == "softmax_cosine":
        kwargs.setdefault("temporal_pool", "softmax")
    return cls(vla_dim, vae_dim, **kwargs)
