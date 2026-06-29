#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# LIBERO single-task evaluation with attention-heatmap visualization.
#
# Differences vs run_libero_eval.sh:
#   * no parallelism — one GPU, one server, one suite, one task.
#   * defaults to a small number of trials (3) so the heatmap output is
#     manageable.
#   * exports the PI0_ATTN_VIS_* env vars so the policy server saves
#     per-camera attention overlays under  $LOG_DIR/attn_vis/
#       └── task<NN>_<task_name>/
#             └── ep<NNN>/
#                   └── step<MMMM>_<branch>_layer<LL>_b0_<camera>_overlay.png
# ─────────────────────────────────────────────────────────────────────────────
set -e

export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

#############################
# Evaluation configuration  (edit me)
#############################
SUITE="libero_10"            # one of: libero_spatial, libero_object, libero_goal, libero_10, libero_90
TASK_ID=9                        # which task index inside the suite (0-based)
NUM_TRIALS=50                      # rollouts of the chosen task
                    # any free port

# Server policy: "default" uses Pi0_fast_libero weights; "custom" uses the checkpoint below.
POLICY_MODE="custom"              # "default" or "custom"


# GPU=0
# PORT=8200     
# POLICY_CONFIG="pi0_libero"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/pi0_libero_original_4gpu/30000"


# GPU=1
# PORT=8201     
# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_wanvae0-8_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"


# GPU=2
# PORT=8202     
# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_dino0-8_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"


# GPU=3
# PORT=8203     
# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_siglip0-8_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"


GPU=4
PORT=8204     
POLICY_CONFIG="pi0_libero_video_layer11_5e-1_vggt0-8_multi_frame_concat"
POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"


#############################
# Attention-visualization configuration  (edit me)
#############################
# Comma-separated layer indices (PaliGemma is 18-layer; deeper layers usually
# more semantic). Each extra layer ~doubles output volume.
ATTN_LAYERS="11"

# How often (in sample_actions calls) to save a snapshot. 1 = every call.
# At replan_steps=5, 1 call ≈ 5 env steps, so SAMPLE_EVERY=2 ≈ every 10 steps.
ATTN_SAMPLE_EVERY=1

# Hard cap on the total number of saved snapshots (None / unset = unlimited).
# Set to a number to bound the run-time / disk usage.
ATTN_MAX_CALLS=6000

# Capture suffix (action-token → image) attention. Recommended.
ATTN_SAVE_SUFFIX=1
# Capture prefix (language-token → image) attention. Off by default.
ATTN_SAVE_PREFIX=0

# Which denoise step to record for the suffix branch ("first"|"middle"|"last"|<int>).
ATTN_STEP="first"

# Heatmap blending factor on top of the original camera frame, [0,1].
ATTN_ALPHA=0.5

# Upper-tail clip percentile for per-image heatmap normalization. Trained
# transformers tend to put a fixed "attention sink" on a couple of patches
# regardless of content, which under naive min-max normalization (=100) shows
# up as a bright blob in the same corner of every image. 99 caps the top 1%
# so the actual grounding signal is visible. Lower (e.g. 95) = even more
# aggressive suppression; 100 = disable clipping.
ATTN_CLIP_PCT=99

# Number of patch RINGS along the border to *consider* for sink suppression.
#   0 = no border processing at all (raw display, sink visible as red ring).
#   1 = check the outermost ring only.
#   2 (default) = check the outermost two rings (PaliGemma SigLIP sinks
#                 typically extend that deep).
ATTN_BORDER_MASK=2

# How to suppress border patches:
#   "fade" (DEFAULT, recommended) — every border patch is multiplied by a
#          smooth cosine ramp that grows from ATTN_BORDER_FADE_MIN at the
#          very outermost edge to 1.0 at depth ATTN_BORDER_MASK patches
#          inward. *Every* border value (sink AND legitimate peak) gets
#          attenuated; the smooth ramp guarantees no visible boundary.
#   "outlier" — only border patches whose value exceeds the
#               ATTN_BORDER_OUTLIER_PCT-th percentile of the interior are
#               capped at that value (legitimate-low borders pass through).
#   "clip"   — clip every border patch down to interior median.
#   "interior_mean" — replace every border patch with interior mean.
#   "zero"   — hard-zero (legacy, may leave a dark frame).
ATTN_BORDER_MODE="fade"

# Outlier threshold percentile (only used when ATTN_BORDER_MODE="outlier"):
#   75 (default) = top 25% of interior values are treated as outliers
#   50           = top 50% (stricter, ~ similar to "clip" mode)
#   90           = top 10% (only most extreme sinks; mild)
ATTN_BORDER_OUTLIER_PCT=75

# Fade-min weight at the very outermost edge (only used when ATTN_BORDER_MODE="fade"):
#   0.0 (default) = outermost ring is fully zeroed, ramps smoothly back up
#   0.3           = keeps 30% of the original value at the edge (mild)
#   1.0           = no attenuation anywhere (effectively disables fade)
ATTN_BORDER_FADE_MIN=0.0

# ATTN_GAMMA: sharpening exponent applied to the normalized heatmap.
#   1.0 (default) = linear, most faithful to data
#   1.2-1.5 = moderate, peaks pop out more, mids slightly suppressed
#   2.0+ = strong, only true hotspots visible
#
# ATTN_MOD_ALPHA: how the heatmap blends with the camera image.
#   1 (default) = per-pixel alpha modulated by heatmap intensity
#   0 = flat alpha everywhere (whole interior tinted blue->red, "classic" look)
#
# ATTN_ALPHA_FLOOR: minimum fraction of ATTN_ALPHA kept in cold regions when
# modulating, in [0, 1]. Controls "background blue tint":
#   0.0  = cold regions show exact original image (and you'll likely see a
#          visible boundary between border and the interior).
#   0.2  = (DEFAULT) faint blue background everywhere, border smoothly blends
#          into cold interior — recommended.
#   0.4  = noticeable blue background, hot regions still pop.
#   1.0  = effectively the same as ATTN_MOD_ALPHA=0 (flat overlay).
#
# Quick presets (copy-paste):
#   "with bg tint" gamma=1.0  mod_alpha=1  alpha_floor=0.2  <-- DEFAULT
#   "more bg"      gamma=1.0  mod_alpha=1  alpha_floor=0.4
#   "hard mask"    gamma=1.0  mod_alpha=1  alpha_floor=0.0  (cold = pure orig)
#   "full overlay" gamma=1.0  mod_alpha=0  alpha_floor=*    (alpha_floor unused)
#   "sharp peaks"  gamma=2.0  mod_alpha=1  alpha_floor=0.1
ATTN_GAMMA=1.0
ATTN_MOD_ALPHA=1
ATTN_ALPHA_FLOOR=0.2

# Cameras to skip (comma-separated, no spaces). The model still runs all
# cameras through attention; we just don't write PNGs for the listed ones.
# Possible names: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb.
# Leave empty to save all three. Example: "left_wrist_0_rgb,right_wrist_0_rgb"
ATTN_SKIP_CAMERAS="left_wrist_0_rgb,right_wrist_0_rgb"

#############################
# Derived paths
#############################
LIBERO_VENV="$SCRIPT_DIR/examples/libero/.venv/bin/activate"
LIBERO_THIRD_PARTY="$SCRIPT_DIR/third_party/libero"

if [ "$POLICY_MODE" = "custom" ]; then
    EXP_NAME="$(basename "$(dirname "$POLICY_DIR")")"
    CKPT_NUM="$(basename "$POLICY_DIR")"
else
    EXP_NAME="default"
    CKPT_NUM="default"
fi

LOG_DIR="logs_vis/${EXP_NAME}/${CKPT_NUM}/${SUITE}/task${TASK_ID}"
ATTN_DIR="${LOG_DIR}/attn_vis"

mkdir -p "$LOG_DIR" "$ATTN_DIR"

SERVER_PID=""
EVAL_PID=""

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "$SERVER_PID" "$EVAL_PID"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null && echo "  Killed PID $pid"
        fi
    done
    wait 2>/dev/null
    echo "All processes stopped."
}
trap cleanup EXIT INT TERM

########################################
# 1. Start a single policy server with attention-vis enabled
########################################
echo "========== Starting policy server (GPU ${GPU}, port ${PORT}) =========="
echo "  Suite        : ${SUITE}"
echo "  Task id      : ${TASK_ID}"
echo "  Trials       : ${NUM_TRIALS}"
echo "  Layers       : ${ATTN_LAYERS}"
echo "  Sample every : ${ATTN_SAMPLE_EVERY}"
echo "  Max snapshots: ${ATTN_MAX_CALLS}"
echo "  Output dir   : ${ATTN_DIR}"

SERVER_CMD_EXTRA=()
if [ "$POLICY_MODE" = "custom" ]; then
    SERVER_CMD_EXTRA=(policy:checkpoint --policy.config "$POLICY_CONFIG" --policy.dir "$POLICY_DIR")
fi

CUDA_VISIBLE_DEVICES=${GPU} \
TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache_gpu${GPU}_vis" \
PI0_ATTN_VIS_LAYERS="${ATTN_LAYERS}" \
PI0_ATTN_VIS_DIR="${ATTN_DIR}" \
PI0_ATTN_VIS_SAMPLE_EVERY="${ATTN_SAMPLE_EVERY}" \
PI0_ATTN_VIS_MAX_CALLS="${ATTN_MAX_CALLS}" \
PI0_ATTN_VIS_SUFFIX="${ATTN_SAVE_SUFFIX}" \
PI0_ATTN_VIS_PREFIX="${ATTN_SAVE_PREFIX}" \
PI0_ATTN_VIS_STEP="${ATTN_STEP}" \
PI0_ATTN_VIS_ALPHA="${ATTN_ALPHA}" \
PI0_ATTN_VIS_CLIP_PCT="${ATTN_CLIP_PCT}" \
PI0_ATTN_VIS_BORDER_MASK="${ATTN_BORDER_MASK}" \
PI0_ATTN_VIS_BORDER_MODE="${ATTN_BORDER_MODE}" \
PI0_ATTN_VIS_OUTLIER_PCT="${ATTN_BORDER_OUTLIER_PCT}" \
PI0_ATTN_VIS_FADE_MIN="${ATTN_BORDER_FADE_MIN}" \
PI0_ATTN_VIS_GAMMA="${ATTN_GAMMA}" \
PI0_ATTN_VIS_MOD_ALPHA="${ATTN_MOD_ALPHA}" \
PI0_ATTN_VIS_ALPHA_FLOOR="${ATTN_ALPHA_FLOOR}" \
PI0_ATTN_VIS_SKIP_CAMERAS="${ATTN_SKIP_CAMERAS}" \
uv run scripts/serve_policy.py \
    --env LIBERO --port "${PORT}" \
    "${SERVER_CMD_EXTRA[@]}" \
    > "${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for server (PID ${SERVER_PID}) to be ready on port ${PORT}..."
for attempt in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1; then
        echo "  Server is ready."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died. See ${LOG_DIR}/server.log"
        exit 1
    fi
    if [ "$attempt" -eq 180 ]; then
        echo "ERROR: Server did not start within 180s. Check ${LOG_DIR}/server.log"
        exit 1
    fi
    sleep 1
done

########################################
# 2. Run a single-task LIBERO evaluation
########################################
echo "========== Running main_vis.py =========="

(
    source "$LIBERO_VENV"
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$LIBERO_THIRD_PARTY"
    python examples/libero/main_vis.py \
        --args.task-suite-name "${SUITE}" \
        --args.task-id "${TASK_ID}" \
        --args.num-trials-per-task "${NUM_TRIALS}" \
        --args.port "${PORT}" \
        --args.video-out-path "${LOG_DIR}/videos"
) > "${LOG_DIR}/eval.log" 2>&1 &
EVAL_PID=$!

echo "Eval launched (PID ${EVAL_PID})."
echo "  Server log    : ${LOG_DIR}/server.log"
echo "  Eval log      : ${LOG_DIR}/eval.log"
echo "  Videos        : ${LOG_DIR}/videos/"
echo "  Attention vis : ${ATTN_DIR}/"

if wait "${EVAL_PID}"; then
    echo "========== Evaluation completed successfully =========="
else
    echo "========== Evaluation failed. See ${LOG_DIR}/eval.log =========="
    exit 1
fi
