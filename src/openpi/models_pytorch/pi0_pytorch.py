import logging
import math
import os
import pathlib

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch import attention_viz as _attn_viz
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # Keep an uncompiled bound reference around so that attention-capture can
        # bypass torch.compile (which does not play well with Python-level file
        # I/O and dynamic dict mutation).
        self._uncompiled_sample_actions = self.sample_actions
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Attention-capture configuration; populated by enable_attention_capture()
        # or via the PI0_ATTN_VIS_* env vars (see _maybe_init_attention_capture_from_env).
        self._attn_cap_cfg: dict | None = None
        self._maybe_init_attention_capture_from_env()

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    # ── Attention visualization API ────────────────────────────────────────
    def enable_attention_capture(
        self,
        *,
        layers,
        save_dir,
        image_keys=None,
        skip_cameras=None,
        max_calls: int | None = None,
        sample_every: int = 1,
        save_prefix: bool = False,
        save_suffix: bool = True,
        alpha: float = 1.0,
        save_originals: bool = True,
    ) -> None:
        """Enable saving of attention heatmaps during ``sample_actions``.

        Args:
            layers: iterable of int, 0-indexed layer indices in the PaliGemma /
                action-expert stack to record.
            save_dir: directory to write PNGs to.
            image_keys: ordered camera names matching the prefix order. Defaults
                to :data:`openpi.models_pytorch.preprocessing_pytorch.IMAGE_KEYS`.
            skip_cameras: iterable of camera names to *exclude* from saving.
                Their attention is still computed (the model always runs all
                cameras), but no PNGs are written for them.
            max_calls: stop capturing after this many *saved* ``sample_actions``
                calls (``None`` means unlimited).
            sample_every: only save once every N ``sample_actions`` calls.
            save_prefix: render heatmaps from prefix self-attention
                (language→image grounding).
            save_suffix: render heatmaps from suffix-to-prefix attention
                (action queries attending to image tokens) at the first
                denoise step.
            alpha: flat blending factor for the heatmap on the image.
            save_originals: also write a copy of the source camera image.

        Rollout-aware folder layout:
            When the client passes ``__rollout_task_id__`` /
            ``__rollout_episode_id__`` / ``__rollout_step__`` / optionally
            ``__rollout_task_name__`` through ``Policy.infer`` (which calls
            :meth:`set_rollout_context` on this model), saved files are
            organized as ``save_dir/task<NN>_<task_name>/ep<NNN>/step<MMMM>_*``.
            Without that metadata, files are written flat under ``save_dir``.
        """
        layers = sorted({int(x) for x in layers})
        if not layers:
            raise ValueError("`layers` must contain at least one layer index")
        if image_keys is None:
            image_keys = list(_preprocessing.IMAGE_KEYS)
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if int(sample_every) < 1:
            raise ValueError("`sample_every` must be >= 1")
        self._attn_cap_cfg = {
            "layers": layers,
            "save_dir": save_dir,
            "image_keys": list(image_keys),
            "skip_cameras": set(skip_cameras) if skip_cameras else set(),
            "max_calls": max_calls,
            "sample_every": int(sample_every),
            "save_prefix": save_prefix,
            "save_suffix": save_suffix,
            "alpha": float(alpha),
            "save_originals": save_originals,
            "call_idx": 0,   # number of sample_actions calls that actually produced files
            "seen_idx": 0,   # total number of sample_actions calls observed
            # Last-seen explicit rollout label (for folder naming).
            "_last_task_label": None,
        }
        # Bypass torch.compile so that the Python-level capture / file I/O run
        # eagerly. Save & rebind only if compile was actually applied.
        if self.sample_actions is not self._uncompiled_sample_actions:
            self.sample_actions = self._uncompiled_sample_actions
            logging.info(
                "Disabled torch.compile on sample_actions because attention capture is enabled."
            )
        logging.info(
            "Enabled attention capture: layers=%s save_dir=%s save_prefix=%s save_suffix=%s",
            layers,
            save_dir,
            save_prefix,
            save_suffix,
        )

    def disable_attention_capture(self) -> None:
        """Disable attention capture."""
        self._attn_cap_cfg = None

    def is_attention_capture_enabled(self) -> bool:
        return self._attn_cap_cfg is not None

    def _maybe_init_attention_capture_from_env(self) -> None:
        """Activate capture if ``PI0_ATTN_VIS_LAYERS`` and ``PI0_ATTN_VIS_DIR`` are set.

        Recognised env vars:
            PI0_ATTN_VIS_LAYERS       - comma-separated layer indices, e.g. "5,10,15"
            PI0_ATTN_VIS_DIR          - output directory
            PI0_ATTN_VIS_MAX_CALLS    - integer, max snapshots to save (default unlimited)
            PI0_ATTN_VIS_SAMPLE_EVERY - save once every N sample_actions calls (default 1)
            PI0_ATTN_VIS_PREFIX       - "1" to also save prefix self-attention
            PI0_ATTN_VIS_SUFFIX       - "0" to skip suffix-to-prefix attention (default on)
            PI0_ATTN_VIS_SKIP_CAMERAS - comma-separated camera names to skip,
                                        e.g. "left_wrist_0_rgb,right_wrist_0_rgb"
        """
        layers_env = os.environ.get("PI0_ATTN_VIS_LAYERS")
        save_dir_env = os.environ.get("PI0_ATTN_VIS_DIR")
        if not layers_env or not save_dir_env:
            return
        try:
            layers = [int(x) for x in layers_env.split(",") if x.strip()]
        except ValueError:
            logging.warning("Ignoring PI0_ATTN_VIS_LAYERS=%r (not comma-separated ints)", layers_env)
            return
        max_calls_env = os.environ.get("PI0_ATTN_VIS_MAX_CALLS")
        max_calls = int(max_calls_env) if max_calls_env else None
        sample_every = int(os.environ.get("PI0_ATTN_VIS_SAMPLE_EVERY", "1"))
        self.enable_attention_capture(
            layers=layers,
            save_dir=save_dir_env,
            max_calls=max_calls,
            sample_every=sample_every,
            save_prefix=os.environ.get("PI0_ATTN_VIS_PREFIX", "0") == "1",
            save_suffix=os.environ.get("PI0_ATTN_VIS_SUFFIX", "1") == "1",
            skip_cameras=[
                x.strip()
                for x in os.environ.get("PI0_ATTN_VIS_SKIP_CAMERAS", "").split(",")
                if x.strip()
            ] or None,
        )

    def set_rollout_context(
        self,
        *,
        task_id: int | None = None,
        episode_id: int | None = None,
        step: int | None = None,
        task_name: str | None = None,
    ) -> None:
        """Provide explicit rollout coordinates for the *next* sample_actions call.

        The PI0 model is otherwise stateless, so the LIBERO eval client can call
        this (typically via ``Policy.infer``'s ``__rollout_*__`` magic keys, see
        :meth:`Policy.infer`) once per step. The values are consumed at the
        start of the next ``sample_actions`` call and override the
        prompt-hash + time-gap heuristic used by default.
        """
        if self._attn_cap_cfg is None:
            return
        self._attn_cap_cfg["_explicit_rollout"] = {
            "task_id": int(task_id) if task_id is not None else None,
            "episode_id": int(episode_id) if episode_id is not None else None,
            "step": int(step) if step is not None else None,
            "task_name": str(task_name) if task_name is not None else None,
        }

    @staticmethod
    def _sanitize_name(name: str, max_len: int = 40) -> str:
        out = "".join(c if c.isalnum() else "_" for c in name).strip("_")
        return out[:max_len] or "task"

    def _update_rollout_context(self) -> tuple[int, int, int] | None:
        """Consume the explicit rollout context set via ``set_rollout_context``.

        Returns ``(task_idx, ep_idx, step_in_ep)`` if the client provided one,
        otherwise ``None`` (in which case files are written flat under
        ``save_dir``). No heuristic fallback is performed.
        """
        cfg = self._attn_cap_cfg
        explicit = cfg.pop("_explicit_rollout", None)
        if explicit is None or explicit.get("task_id") is None:
            return None
        task_idx = int(explicit["task_id"])
        ep_idx = int(explicit.get("episode_id") or 0)
        step_in_ep = int(explicit["step"]) if explicit.get("step") is not None else 0
        label = explicit.get("task_name")
        cfg["_last_task_label"] = self._sanitize_name(label) if label else None
        return task_idx, ep_idx, step_in_ep

    def _rollout_subdir(self, task_idx: int, ep_idx: int) -> pathlib.Path:
        cfg = self._attn_cap_cfg
        base = cfg["save_dir"]
        label = cfg.get("_last_task_label")
        task_dir = base / (f"task{task_idx:02d}_{label}" if label else f"task{task_idx:02d}")
        return task_dir / f"ep{ep_idx:03d}"

    def _save_attention_heatmaps(
        self,
        *,
        orig_images: dict,
        prefix_collector: dict | None,
        suffix_collector: dict | None,
        prefix_len: int,
        lang_len: int,
        num_cameras: int,
        rollout_ctx: tuple[int, int, int] | None = None,
    ) -> None:
        cfg = self._attn_cap_cfg
        image_len = prefix_len - lang_len
        if num_cameras <= 0 or image_len <= 0 or image_len % num_cameras != 0:
            logging.warning(
                "Skipping attention heatmap save: prefix layout looks unexpected "
                "(prefix_len=%d, lang_len=%d, num_cameras=%d).",
                prefix_len, lang_len, num_cameras,
            )
            return
        tokens_per_cam = image_len // num_cameras
        side = int(math.isqrt(tokens_per_cam))
        if side * side != tokens_per_cam:
            logging.warning(
                "Skipping attention heatmap save: tokens_per_cam=%d is not a perfect square.",
                tokens_per_cam,
            )
            return

        seen_idx = cfg["seen_idx"] - 1  # absolute sample_actions call index
        # Determine where to save: rollout-aware subdir if we have a context,
        # otherwise the flat base dir.
        if rollout_ctx is not None:
            task_idx, ep_idx, step_in_ep = rollout_ctx
            save_dir = self._rollout_subdir(task_idx, ep_idx)
            save_dir.mkdir(parents=True, exist_ok=True)
            step_tag = f"step{step_in_ep:04d}"
        else:
            save_dir = cfg["save_dir"]
            step_tag = f"step{seen_idx:06d}"

        if prefix_collector is not None:
            # Use the language-token rows of the prefix self-attention so the
            # heatmap reflects language→image grounding.
            prefix_qslice = slice(image_len, image_len + lang_len) if lang_len > 0 else None
            for layer_idx, attn in prefix_collector.get("prefix", {}).items():
                cam_maps = _attn_viz.split_attention_per_camera(
                    attn,
                    num_cameras=num_cameras,
                    tokens_per_cam=tokens_per_cam,
                    query_slice=prefix_qslice,
                )
                _attn_viz.save_camera_heatmaps(
                    images=orig_images,
                    image_keys=cfg["image_keys"],
                    cam_attn_maps=cam_maps,
                    save_dir=save_dir,
                    tag=f"{step_tag}_prefix_layer{layer_idx:02d}",
                    alpha=cfg["alpha"],
                    save_originals=cfg["save_originals"],
                )

        if suffix_collector is not None:
            # Only the action-token queries attend to image tokens meaningfully.
            query_slice = slice(-self.config.action_horizon, None)
            for layer_idx, attn in suffix_collector.get("suffix", {}).items():
                cam_maps = _attn_viz.split_attention_per_camera(
                    attn,
                    num_cameras=num_cameras,
                    tokens_per_cam=tokens_per_cam,
                    query_slice=query_slice,
                )
                _attn_viz.save_camera_heatmaps(
                    images=orig_images,
                    image_keys=cfg["image_keys"],
                    cam_attn_maps=cam_maps,
                    save_dir=save_dir,
                    tag=f"{step_tag}_suffix_layer{layer_idx:02d}",
                    alpha=cfg["alpha"],
                    save_originals=cfg["save_originals"],
                )

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None,
                video_align_proj=None, vla_align_layer: int = -1,
                use_spatial_pe: bool = False,
                video_cache_layout: str = "CTHW") -> Tensor:
        """Do a full training forward pass and compute the loss.

        When ``video_align_proj`` is provided and the observation contains ``vae_cache``,
        also computes a video alignment loss and returns ``(action_loss, video_align_loss)``.
        Otherwise returns the per-element action MSE loss tensor (original behaviour).
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        use_video_align = video_align_proj is not None

        if use_video_align:
            _align_layer = vla_align_layer if vla_align_layer >= 0 else None

            def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
                (prefix_out, suffix_out), _, intermediate = self.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                    return_prefix_at_layer=_align_layer,
                )
                return prefix_out, suffix_out, intermediate

            prefix_out, suffix_out, intermediate_hidden = self._apply_checkpoint(
                forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
            )
        else:
            def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
                (_, suffix_out), _, _ = self.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                )
                return suffix_out

            suffix_out = self._apply_checkpoint(
                forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
            )
            prefix_out = None

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        action_loss = F.mse_loss(u_t, v_t, reduction="none")

        if not use_video_align:
            return action_loss

        # ── Video alignment branch (pre-computed cache only) ─────────────────
        vae_cache = getattr(observation, "vae_cache", None)
        if vae_cache is None:
            return action_loss

        # vae_cache: [B, N_vae_cams, C, T', H', W']  (CTHW layout)
        vae_cache = vae_cache.to(actions.device)
        if video_cache_layout == "TCHW":
            # [B, N_vae_cams, T, C, H, W] -> [B, N_vae_cams, C, T, H, W]
            vae_cache = vae_cache.permute(0, 1, 3, 2, 4, 5).contiguous()
        num_cameras = len(images)
        img_len = prefix_embs.shape[1] - lang_masks.shape[1]
        tokens_per_cam = img_len // num_cameras
        n_vae_cams = vae_cache.shape[1]

        valid_cam_indices = [i for i, m in enumerate(img_masks) if m.any()]

        align_hidden = intermediate_hidden if intermediate_hidden is not None else prefix_out

        cam_vla_tokens = []
        cam_vae_tokens = []
        mask_parts = []

        _proj_mod = video_align_proj.module if hasattr(video_align_proj, "module") else video_align_proj
        multi_frame = getattr(_proj_mod, "multi_frame", False)
        is_dual_head = getattr(_proj_mod, "dual_head", False)

        for vae_idx, vla_cam_idx in enumerate(valid_cam_indices[:n_vae_cams]):
            cam_start = vla_cam_idx * tokens_per_cam
            cam_end = cam_start + tokens_per_cam
            cam_vla_tokens.append(align_hidden[:, cam_start:cam_end, :])
            mask_parts.append(prefix_pad_masks[:, cam_start:cam_end])
            cam_latent = vae_cache[:, vae_idx]  # [B, 48, T', H', W']
            if multi_frame:
                cam_vae_tokens.append(self._match_vae_to_vla_grid_multiframe(cam_latent, tokens_per_cam, use_spatial_pe))
            else:
                cam_vae_tokens.append(self._match_vae_to_vla_grid(cam_latent, tokens_per_cam, use_spatial_pe))

        vision_hidden = torch.cat(cam_vla_tokens, dim=1)
        vae_tokens = torch.cat(cam_vae_tokens, dim=1)
        img_mask = torch.cat(mask_parts, dim=1)

        # Prepare auxiliary VAE tokens for dual-head projectors
        vae_tokens_aux = None
        vae_cache_aux = getattr(observation, "vae_cache_aux", None)
        if is_dual_head and vae_cache_aux is not None:
            vae_cache_aux = vae_cache_aux.to(actions.device)
            if video_cache_layout == "TCHW":
                vae_cache_aux = vae_cache_aux.permute(0, 1, 3, 2, 4, 5).contiguous()
            cam_vae_tokens_aux = []
            n_vae_cams_aux = vae_cache_aux.shape[1]
            for vae_idx, vla_cam_idx in enumerate(valid_cam_indices[:n_vae_cams_aux]):
                cam_latent_aux = vae_cache_aux[:, vae_idx]
                cam_vae_tokens_aux.append(
                    self._match_vae_to_vla_grid_multiframe(cam_latent_aux, tokens_per_cam, use_spatial_pe)
                )
            vae_tokens_aux = torch.cat(cam_vae_tokens_aux, dim=1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            proj_kwargs = {}
            if vae_tokens_aux is not None:
                proj_kwargs["vae_hidden_aux"] = vae_tokens_aux.float()
            video_align_loss = video_align_proj(
                vision_hidden.float(), vae_tokens.float(), img_mask,
                **proj_kwargs,
            )

        return action_loss, video_align_loss

    # ── VAE-to-VLA alignment helpers ────────────────────────────────────────

    @staticmethod
    def _match_vae_to_vla_grid(vae_latent: Tensor, num_vla_tokens: int,
                                use_spatial_pe: bool = False) -> Tensor:
        """Reshape VAE latent ``[B, C, T', H', W']`` to match VLA spatial token grid ``[B, N, C]``."""
        spatial = vae_latent.mean(dim=2)  # [B, C, H', W']
        if use_spatial_pe:
            from openpi.models_pytorch.video_projector import apply_spatial_pe
            spatial = apply_spatial_pe(spatial, ratio=0.1)
        target_side = int(num_vla_tokens ** 0.5)
        if spatial.shape[2] != target_side or spatial.shape[3] != target_side:
            spatial = F.interpolate(
                spatial, size=(target_side, target_side), mode="bilinear", align_corners=False
            )
        return spatial.flatten(2).permute(0, 2, 1)  # [B, N, C]

    @staticmethod
    def _match_vae_to_vla_grid_multiframe(vae_latent: Tensor, num_vla_tokens: int,
                                           use_spatial_pe: bool = False) -> Tensor:
        """Reshape VAE latent ``[B, C, T', H', W']`` to ``[B, N, T', C]``."""
        B, C, T, H, W = vae_latent.shape
        if use_spatial_pe:
            from openpi.models_pytorch.video_projector import apply_spatial_pe
            flat = vae_latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            flat = apply_spatial_pe(flat, ratio=0.1)
            vae_latent = flat.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        target_side = int(num_vla_tokens ** 0.5)
        if H != target_side or W != target_side:
            flat = vae_latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            flat = F.interpolate(flat, size=(target_side, target_side), mode="bilinear", align_corners=False)
            vae_latent = flat.reshape(B, T, C, target_side, target_side).permute(0, 2, 1, 3, 4)
        # [B, C, T, h, w] -> [B, N, T, C]  where N = h*w
        return vae_latent.flatten(3).permute(0, 3, 2, 1)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Snapshot original camera images (post-resize, pre-augment) for overlays.
        attn_cfg = self._attn_cap_cfg
        capture_active = False
        rollout_ctx: tuple[int, int, int] | None = None
        if attn_cfg is not None:
            # Always update rollout context so the task/ep counters track every
            # call, not only the strided ones.
            rollout_ctx = self._update_rollout_context()
            seen = attn_cfg["seen_idx"]
            attn_cfg["seen_idx"] = seen + 1
            stride = attn_cfg.get("sample_every", 1)
            max_calls = attn_cfg.get("max_calls")
            if (seen % stride == 0) and (max_calls is None or attn_cfg["call_idx"] < max_calls):
                capture_active = True
                attn_cfg["call_idx"] += 1

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        orig_images: dict | None = None
        if capture_active:
            # Use the preprocessed (resized to 224x224) images so that the heatmap
            # grid aligns exactly with what the vision encoder consumed. Cameras
            # listed in ``skip_cameras`` are silently dropped here so the
            # downstream saver naturally writes nothing for them.
            cfg_keys = attn_cfg["image_keys"]
            skip = attn_cfg.get("skip_cameras", set())
            orig_images = {}
            for idx, img in enumerate(images):
                key = cfg_keys[idx] if idx < len(cfg_keys) else f"cam_{idx}"
                if key in skip:
                    continue
                orig_images[key] = img.detach()

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        prefix_collector: dict | None = None
        if capture_active and attn_cfg["save_prefix"]:
            prefix_collector = {"layers": attn_cfg["layers"]}

        _, past_key_values, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            attn_collector=prefix_collector,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        # Always capture the first denoise step when suffix capture is enabled.
        suffix_capture_step = 0 if capture_active and attn_cfg["save_suffix"] else -1
        suffix_collector: dict | None = None

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        step_idx = 0
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            cur_collector: dict | None = None
            if step_idx == suffix_capture_step:
                cur_collector = {"layers": attn_cfg["layers"]}
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                attn_collector=cur_collector,
            )
            if cur_collector is not None:
                suffix_collector = cur_collector

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
            step_idx += 1

        if capture_active:
            try:
                self._save_attention_heatmaps(
                    orig_images=orig_images or {},
                    prefix_collector=prefix_collector,
                    suffix_collector=suffix_collector,
                    prefix_len=prefix_pad_masks.shape[1],
                    lang_len=lang_masks.shape[1],
                    num_cameras=len(images),
                    rollout_ctx=rollout_ctx,
                )
            except Exception:  # pragma: no cover - never break inference on viz error
                logging.exception("Failed to save attention heatmaps; continuing.")

        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        attn_collector: dict | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            attn_collector=attn_collector,
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
