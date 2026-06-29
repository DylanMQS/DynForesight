"""Quick inference-speed benchmark for a trained openpi checkpoint.

Loads the policy in-process (no websocket server) and runs many synthetic
LIBERO-style observations through `policy.infer()`. Reports per-call latency
statistics. Designed for the PyTorch checkpoint at e.g.
``checkpoints/pi05_libero/pi05_libero_original_4gpu/30000``.

Example:
    uv run scripts/benchmark_inference.py \
        --config pi05_libero \
        --checkpoint checkpoints/pi05_libero/pi05_libero_original_4gpu/30000 \
        --warmup 5 --iters 50
"""
from __future__ import annotations

import dataclasses
import logging
import statistics
import time

import numpy as np
import tyro

from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str = "pi05_libero"
    checkpoint: str = "checkpoints/pi05_libero/pi05_libero_original_4gpu/30000"
    warmup: int = 5
    iters: int = 50
    image_size: int = 224
    state_dim: int = 8
    prompt: str = "pick up the black bowl and place it on the plate"
    device: str | None = None  # e.g. "cuda" / "cuda:0" / "cpu"; None -> auto
    seed: int = 0


def _make_obs(image_size: int, state_dim: int, prompt: str, rng: np.random.Generator) -> dict:
    """Mimics the dict produced by examples/libero/main.py before client.infer()."""
    img = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
    state = rng.standard_normal(state_dim).astype(np.float32)
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }


def _summarize(label: str, samples_ms: list[float]) -> None:
    n = len(samples_ms)
    avg = statistics.mean(samples_ms)
    med = statistics.median(samples_ms)
    mn = min(samples_ms)
    mx = max(samples_ms)
    std = statistics.stdev(samples_ms) if n > 1 else 0.0
    hz = 1000.0 / avg if avg > 0 else float("inf")
    print(
        f"[{label}] n={n:3d} | mean={avg:8.2f} ms | median={med:8.2f} ms | "
        f"min={mn:8.2f} ms | max={mx:8.2f} ms | std={std:7.2f} ms | ~{hz:6.2f} Hz"
    )


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    print(f"Loading config '{args.config}' and checkpoint '{args.checkpoint}'...")
    train_config = _config.get_config(args.config)
    t0 = time.monotonic()
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        default_prompt=args.prompt,
        pytorch_device=args.device,
    )
    load_s = time.monotonic() - t0
    print(f"Policy loaded in {load_s:.1f}s (device={getattr(policy, '_pytorch_device', 'jax')}).")
    print(
        f"Model: action_horizon={train_config.model.action_horizon}, "
        f"action_dim={train_config.model.action_dim}, "
        f"max_token_len={train_config.model.max_token_len}"
    )

    rng = np.random.default_rng(args.seed)
    sample = _make_obs(args.image_size, args.state_dim, args.prompt, rng)

    print(f"\n--- Warmup x{args.warmup} (first call may compile, expect long latency) ---")
    warmup_ms: list[float] = []
    for i in range(args.warmup):
        obs = _make_obs(args.image_size, args.state_dim, args.prompt, rng)
        t0 = time.monotonic()
        out = policy.infer(obs)
        dt = (time.monotonic() - t0) * 1000.0
        warmup_ms.append(dt)
        infer_ms = out.get("policy_timing", {}).get("infer_ms")
        print(f"  warmup {i+1}/{args.warmup}: total={dt:9.2f} ms | model={infer_ms:9.2f} ms"
              f" | actions.shape={np.asarray(out['actions']).shape}")

    print(f"\n--- Timed loop x{args.iters} ---")
    total_ms: list[float] = []
    model_ms: list[float] = []
    for i in range(args.iters):
        obs = _make_obs(args.image_size, args.state_dim, args.prompt, rng)
        t0 = time.monotonic()
        out = policy.infer(obs)
        dt = (time.monotonic() - t0) * 1000.0
        total_ms.append(dt)
        m = out.get("policy_timing", {}).get("infer_ms")
        if m is not None:
            model_ms.append(float(m))
        if (i + 1) % max(1, args.iters // 10) == 0:
            print(f"  iter {i+1:3d}/{args.iters}: total={dt:8.2f} ms | model={m:8.2f} ms")

    print()
    _summarize("total (incl. transforms)", total_ms)
    if model_ms:
        _summarize("model.sample_actions   ", model_ms)

    actions = np.asarray(out["actions"])
    print(
        f"\nLast action chunk: shape={actions.shape}, dtype={actions.dtype}, "
        f"min={actions.min():.3f}, max={actions.max():.3f}, mean={actions.mean():.3f}"
    )
    # If we predict H actions per inference but only execute K of them, effective control
    # frequency is roughly (action_horizon / K) * (1 / mean_latency).
    if total_ms:
        mean_s = (sum(total_ms) / len(total_ms)) / 1000.0
        ah = train_config.model.action_horizon
        for replan in (1, 5, 10, ah):
            if replan <= 0 or replan > ah:
                continue
            ctrl_hz = (ah / replan) / mean_s
            print(
                f"  effective control rate @ replan_steps={replan:>3d} (chunk size {ah}): "
                f"~{ctrl_hz:7.2f} Hz"
            )


if __name__ == "__main__":
    main(tyro.cli(Args))
