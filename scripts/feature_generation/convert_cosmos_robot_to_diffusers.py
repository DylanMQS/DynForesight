#!/usr/bin/env python3
"""Convert Cosmos robot/multiview-agibot native .pt checkpoint to diffusers format.

Downloads the NVIDIA native checkpoint from HuggingFace, extracts the DiT weights,
maps them to the diffusers ``CosmosTransformer3DModel`` key convention, and saves
a diffusers-compatible checkpoint directory that can be used as a drop-in replacement
in ``precompute_cosmos_dit_features.py``.

The robot/multiview-agibot model is structurally identical to the base model except
for an extra ``cam_encoder`` (nn.Linear(1536, 2048)) in each block, which is discarded
here since it requires Plücker-ray camera inputs we don't have.

Usage:
    python scripts/convert_cosmos_robot_to_diffusers.py \
        --pt-path /path/to/f740321e-..._ema_bf16.pt \
        --output-dir assets/cosmos_robot_multiview_diffusers

    # Then use it in precompute_cosmos_dit_features.py:
    python scripts/precompute_cosmos_dit_features.py \
        --config-name pi05_aloha_video_align \
        --transformer-path assets/cosmos_robot_multiview_diffusers \
        --revision-tag robotmv
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _build_key_mapping(num_blocks: int = 28):
    """Build the NVIDIA-native → diffusers key mapping for the full DiT.

    Returns a list of (native_key_pattern, diffusers_key_or_callable) tuples.
    Patterns are applied in order; first match wins.

    Native hierarchy (under ``net.``):
        blocks.{i}.self_attn.{q,k,v}_proj.weight  →  transformer_blocks.{i}.attn1.to_{q,k,v}.weight
        blocks.{i}.self_attn.{q,k}_norm.weight     →  transformer_blocks.{i}.attn1.norm_{q,k}.weight
        blocks.{i}.self_attn.output_proj.weight     →  transformer_blocks.{i}.attn1.to_out.0.weight
        blocks.{i}.cross_attn.q_proj.weight         →  transformer_blocks.{i}.attn2.to_q.weight
        blocks.{i}.cross_attn.{k,v}_proj.weight     →  transformer_blocks.{i}.attn2.to_{k,v}.weight
        blocks.{i}.cross_attn.{q,k}_norm.weight     →  transformer_blocks.{i}.attn2.norm_{q,k}.weight
        blocks.{i}.cross_attn.output_proj.weight     →  transformer_blocks.{i}.attn2.to_out.0.weight
        blocks.{i}.mlp.layer1.weight                →  transformer_blocks.{i}.ff.net.0.proj.weight
        blocks.{i}.mlp.layer2.weight                →  transformer_blocks.{i}.ff.net.2.weight
        blocks.{i}.adaln_modulation_self_attn.{1,2}.weight  →  transformer_blocks.{i}.norm1.linear_{1,2}.weight
        blocks.{i}.adaln_modulation_cross_attn.{1,2}.weight →  transformer_blocks.{i}.norm2.linear_{1,2}.weight
        blocks.{i}.adaln_modulation_mlp.{1,2}.weight        →  transformer_blocks.{i}.norm3.linear_{1,2}.weight
        x_embedder.proj.weight                      →  patch_embed.proj.weight
        t_embedder.1.linear_1.weight                →  time_embed.t_embedder.linear_1.weight
        t_embedder.1.linear_2.weight                →  time_embed.t_embedder.linear_2.weight
        t_embedding_norm.weight                     →  time_embed.norm.weight
        crossattn_proj.0.weight                     →  crossattn_proj.0.weight
        crossattn_proj.0.bias                       →  crossattn_proj.0.bias
        final_layer.adaln.1.weight                  →  norm_out.linear_1.weight
        final_layer.adaln.2.weight                  →  norm_out.linear_2.weight
        final_layer.proj_out.weight                 →  proj_out.weight
    """
    mapping = OrderedDict()

    for i in range(num_blocks):
        pfx_n = f"blocks.{i}"
        pfx_d = f"transformer_blocks.{i}"

        # Self-attention
        mapping[f"{pfx_n}.self_attn.q_proj.weight"] = f"{pfx_d}.attn1.to_q.weight"
        mapping[f"{pfx_n}.self_attn.k_proj.weight"] = f"{pfx_d}.attn1.to_k.weight"
        mapping[f"{pfx_n}.self_attn.v_proj.weight"] = f"{pfx_d}.attn1.to_v.weight"
        mapping[f"{pfx_n}.self_attn.q_norm.weight"] = f"{pfx_d}.attn1.norm_q.weight"
        mapping[f"{pfx_n}.self_attn.k_norm.weight"] = f"{pfx_d}.attn1.norm_k.weight"
        mapping[f"{pfx_n}.self_attn.output_proj.weight"] = f"{pfx_d}.attn1.to_out.0.weight"

        # Cross-attention
        mapping[f"{pfx_n}.cross_attn.q_proj.weight"] = f"{pfx_d}.attn2.to_q.weight"
        mapping[f"{pfx_n}.cross_attn.k_proj.weight"] = f"{pfx_d}.attn2.to_k.weight"
        mapping[f"{pfx_n}.cross_attn.v_proj.weight"] = f"{pfx_d}.attn2.to_v.weight"
        mapping[f"{pfx_n}.cross_attn.q_norm.weight"] = f"{pfx_d}.attn2.norm_q.weight"
        mapping[f"{pfx_n}.cross_attn.k_norm.weight"] = f"{pfx_d}.attn2.norm_k.weight"
        mapping[f"{pfx_n}.cross_attn.output_proj.weight"] = f"{pfx_d}.attn2.to_out.0.weight"

        # MLP
        mapping[f"{pfx_n}.mlp.layer1.weight"] = f"{pfx_d}.ff.net.0.proj.weight"
        mapping[f"{pfx_n}.mlp.layer2.weight"] = f"{pfx_d}.ff.net.2.weight"

        # AdaLN modulation (LoRA style: SiLU → Linear(2048,256) → Linear(256, 6144))
        mapping[f"{pfx_n}.adaln_modulation_self_attn.1.weight"] = f"{pfx_d}.norm1.linear_1.weight"
        mapping[f"{pfx_n}.adaln_modulation_self_attn.2.weight"] = f"{pfx_d}.norm1.linear_2.weight"
        mapping[f"{pfx_n}.adaln_modulation_cross_attn.1.weight"] = f"{pfx_d}.norm2.linear_1.weight"
        mapping[f"{pfx_n}.adaln_modulation_cross_attn.2.weight"] = f"{pfx_d}.norm2.linear_2.weight"
        mapping[f"{pfx_n}.adaln_modulation_mlp.1.weight"] = f"{pfx_d}.norm3.linear_1.weight"
        mapping[f"{pfx_n}.adaln_modulation_mlp.2.weight"] = f"{pfx_d}.norm3.linear_2.weight"

    # Non-block components
    mapping["x_embedder.proj.1.weight"] = "patch_embed.proj.weight"
    mapping["t_embedder.1.linear_1.weight"] = "time_embed.t_embedder.linear_1.weight"
    mapping["t_embedder.1.linear_2.weight"] = "time_embed.t_embedder.linear_2.weight"
    mapping["t_embedding_norm.weight"] = "time_embed.norm.weight"
    mapping["crossattn_proj.0.weight"] = "crossattn_proj.0.weight"
    mapping["crossattn_proj.0.bias"] = "crossattn_proj.0.bias"
    mapping["final_layer.adaln_modulation.1.weight"] = "norm_out.linear_1.weight"
    mapping["final_layer.adaln_modulation.2.weight"] = "norm_out.linear_2.weight"
    mapping["final_layer.linear.weight"] = "proj_out.weight"

    return mapping


def convert(pt_path: str, output_dir: str, base_model_id: str, base_revision: str):
    logging.info(f"Loading native checkpoint: {pt_path}")
    native_sd = torch.load(pt_path, map_location="cpu", weights_only=False)

    # The .pt may contain the full model state_dict.
    # DiT keys are prefixed with "net." — extract them.
    net_keys = [k for k in native_sd if k.startswith("net.")]
    if net_keys:
        logging.info(f"Found {len(net_keys)} keys with 'net.' prefix (full model checkpoint).")
        native_dit = {k[len("net."):]: v for k, v in native_sd.items() if k.startswith("net.")}
    else:
        logging.info("No 'net.' prefix found — assuming checkpoint contains only DiT weights.")
        native_dit = native_sd

    logging.info(f"Native DiT has {len(native_dit)} keys.")

    # Build mapping
    mapping = _build_key_mapping(num_blocks=28)

    # Convert
    diffusers_sd = OrderedDict()
    mapped_native_keys = set()
    skipped_keys = []

    for native_key, diffusers_key in mapping.items():
        if native_key in native_dit:
            diffusers_sd[diffusers_key] = native_dit[native_key]
            mapped_native_keys.add(native_key)
        else:
            logging.warning(f"Expected native key not found: {native_key}")

    # Report skipped (unmapped) native keys
    for k in sorted(native_dit.keys()):
        if k not in mapped_native_keys:
            skipped_keys.append(k)

    if skipped_keys:
        logging.info(f"Skipped {len(skipped_keys)} native keys (cam_encoder, LayerNorm, pos_embedder, etc.):")
        cam_keys = [k for k in skipped_keys if "cam_encoder" in k]
        other_keys = [k for k in skipped_keys if "cam_encoder" not in k]
        if cam_keys:
            logging.info(f"  cam_encoder keys (expected, discarded): {len(cam_keys)}")
        for k in other_keys:
            logging.info(f"  {k}  {tuple(native_dit[k].shape)}")

    logging.info(f"Converted {len(diffusers_sd)} keys to diffusers format.")

    # Load the base diffusers model to validate shapes and get config
    logging.info(f"Loading base diffusers model for validation: {base_model_id} revision={base_revision}")
    from diffusers import CosmosTransformer3DModel

    base_model = CosmosTransformer3DModel.from_pretrained(
        base_model_id,
        subfolder="transformer",
        revision=base_revision,
        torch_dtype=torch.bfloat16,
    )
    base_sd = base_model.state_dict()

    # Validate: check that all diffusers keys exist and shapes match
    missing_in_converted = []
    shape_mismatches = []
    for key in base_sd:
        if key not in diffusers_sd:
            missing_in_converted.append(key)
        elif base_sd[key].shape != diffusers_sd[key].shape:
            shape_mismatches.append((key, base_sd[key].shape, diffusers_sd[key].shape))

    extra_in_converted = [k for k in diffusers_sd if k not in base_sd]

    if missing_in_converted:
        logging.warning(f"{len(missing_in_converted)} keys missing in converted (will keep base weights):")
        for k in missing_in_converted:
            logging.warning(f"  {k}")
            diffusers_sd[k] = base_sd[k]

    if shape_mismatches:
        logging.error(f"{len(shape_mismatches)} shape mismatches (KEEPING BASE WEIGHTS for these):")
        for k, base_shape, conv_shape in shape_mismatches:
            logging.error(f"  {k}: base={base_shape}, converted={conv_shape}")
            diffusers_sd[k] = base_sd[k]

    if extra_in_converted:
        logging.warning(f"{len(extra_in_converted)} extra keys in converted (removing):")
        for k in extra_in_converted:
            logging.warning(f"  {k}")
            del diffusers_sd[k]

    # Load into model to verify
    info = base_model.load_state_dict(diffusers_sd, strict=True)
    logging.info(f"load_state_dict result: {info}")

    # Save in diffusers format
    os.makedirs(output_dir, exist_ok=True)
    base_model.save_pretrained(output_dir)
    logging.info(f"Saved diffusers-format transformer to: {output_dir}")
    logging.info(f"")
    logging.info(f"To use with precompute_cosmos_dit_features.py:")
    logging.info(f"  python scripts/precompute_cosmos_dit_features.py \\")
    logging.info(f"      --config-name <your_config> \\")
    logging.info(f"      --transformer-path {os.path.abspath(output_dir)} \\")
    logging.info(f"      --revision-tag robotmv")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Cosmos robot/multiview-agibot .pt to diffusers CosmosTransformer3DModel format."
    )
    parser.add_argument("--pt-path", required=True,
                        help="Path to the native .pt checkpoint (e.g. f740321e-..._ema_bf16.pt)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for the diffusers-format transformer")
    parser.add_argument("--base-model-id", default="nvidia/Cosmos-Predict2.5-2B",
                        help="HuggingFace model id for the base model (used to get config and validate)")
    parser.add_argument("--base-revision", default="diffusers/base/post-trained",
                        help="HuggingFace revision for the base model")
    args = parser.parse_args()

    if not os.path.isfile(args.pt_path):
        print(f"Error: {args.pt_path} not found.", file=sys.stderr)
        print(f"Download it from: https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/tree/main/robot/multiview-agibot",
              file=sys.stderr)
        sys.exit(1)

    convert(args.pt_path, args.output_dir, args.base_model_id, args.base_revision)


if __name__ == "__main__":
    main()
