#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# LIBERO single-task visualization — but for MULTIPLE checkpoints in parallel.
#
# Each row of the (GPUS / PORTS / POLICY_CONFIGS / POLICY_DIRS) arrays is one
# independent server+client pair on its own GPU and port, all evaluating the
# same SUITE / TASK_ID. Heatmaps land in
#   logs_vis/<EXP_NAME_per_config>/30000/<SUITE>/task<TASK_ID>/attn_vis/...
# so different configs never collide.
#
# To skip a row, comment out the same line in *all four* arrays.
# ─────────────────────────────────────────────────────────────────────────────
set -e

export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

#############################
# What to evaluate (shared across all configs)
#############################
SUITE="libero_10"         # one of: libero_spatial, libero_object, libero_goal, libero_10, libero_90
# Which task indices inside the suite (0-based). One or more.
# A single server per config handles all of them; main_vis.py iterates the list
# and the server partitions attn_vis/ by task internally
# (attn_vis/task<NN>_<name>/ep<NNN>/...). Replay videos are filename-tagged by
# task_id so they can safely share videos/.
# Can also be overridden from the CLI, e.g.:
#     ./run_libero_visual_eval_parallel.sh 0 3 8
TASK_IDS=(2 8)
if [ "$#" -gt 0 ]; then
    TASK_IDS=("$@")
fi
NUM_TRIALS=5              # rollouts of each chosen task (per config)

POLICY_MODE="custom"      # "default" or "custom"

#############################
# Per-config arrays (must be the same length; row i = one parallel job)
#############################

GPUS=(
    0
    1
)
PORTS=(
    8200
    8201
)
POLICY_CONFIGS=(
    "pi0_libero"
    "pi0_libero_video_layer11_5e-1_wanvae0-8_multi_frame_concat"
)
POLICY_DIRS=(
    "checkpoints/pi0_libero/pi0_libero_original_4gpu/30000"
    "checkpoints/pi0_libero_video_layer11_5e-1_wanvae0-8_multi_frame_concat/pi0_libero_video_layer11_5e-1_wanvae0-8_multi_frame_concat_4gpu/30000"
)

# GPUS=(
#     0
#     1
#     2
#     3
#     4
# )
# PORTS=(
#     8200
#     8201
#     8202
#     8203
#     8204
# )
# POLICY_CONFIGS=(
#     "pi0_libero"
#     "pi0_libero_video_layer11_5e-1_wanvae0-8_multi_frame_concat"
#     "pi0_libero_video_layer11_5e-1_dino0-8_multi_frame_concat"
#     "pi0_libero_video_layer11_5e-1_siglip0-8_multi_frame_concat"
#     "pi0_libero_video_layer11_5e-1_vggt0-8_multi_frame_concat"
# )
# POLICY_DIRS=(
#     "checkpoints/pi0_libero/pi0_libero_original_4gpu/30000"
#     "checkpoints/pi0_libero_video_layer11_5e-1_wanvae0-8_multi_frame_concat/pi0_libero_video_layer11_5e-1_wanvae0-8_multi_frame_concat_4gpu/30000"
#     "checkpoints/pi0_libero_video_layer11_5e-1_dino0-8_multi_frame_concat/pi0_libero_video_layer11_5e-1_dino0-8_multi_frame_concat_4gpu/30000"
#     "checkpoints/pi0_libero_video_layer11_5e-1_siglip0-8_multi_frame_concat/pi0_libero_video_layer11_5e-1_siglip0-8_multi_frame_concat_4gpu/30000"
#     "checkpoints/pi0_libero_video_layer11_5e-1_vggt0-8_multi_frame_concat/pi0_libero_video_layer11_5e-1_vggt0-8_multi_frame_concat_4gpu/30000"
# )

#############################
# Attention-visualization configuration (shared across all configs)
#############################
ATTN_LAYERS="10,11,12,13,14,15"
ATTN_SAMPLE_EVERY=1
ATTN_MAX_CALLS=1000000
ATTN_SAVE_SUFFIX=1
ATTN_SAVE_PREFIX=0
ATTN_SKIP_CAMERAS="left_wrist_0_rgb,right_wrist_0_rgb"

#############################
# Sanity checks + derived paths
#############################
N=${#GPUS[@]}
if [ "${#PORTS[@]}" -ne "$N" ] || [ "${#POLICY_CONFIGS[@]}" -ne "$N" ] || [ "${#POLICY_DIRS[@]}" -ne "$N" ]; then
    echo "ERROR: GPUS / PORTS / POLICY_CONFIGS / POLICY_DIRS must have the same length."
    echo "  GPUS=${#GPUS[@]}  PORTS=${#PORTS[@]}  CONFIGS=${#POLICY_CONFIGS[@]}  DIRS=${#POLICY_DIRS[@]}"
    exit 1
fi
TASKS_TAG="$(IFS=_; echo "${TASK_IDS[*]}")"   # e.g. "8" or "0_3_8" — used for log path
echo "Will launch ${N} parallel server+client pair(s) for SUITE=${SUITE} TASK_IDS=(${TASK_IDS[*]})"

LIBERO_VENV="$SCRIPT_DIR/examples/libero/.venv/bin/activate"
LIBERO_THIRD_PARTY="$SCRIPT_DIR/third_party/libero"

SERVER_PIDS=()
EVAL_PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${SERVER_PIDS[@]}" "${EVAL_PIDS[@]}"; do
        kill "$pid" 2>/dev/null && echo "  Killed PID $pid"
    done
    wait 2>/dev/null
    echo "All processes stopped."
}
trap cleanup EXIT INT TERM

########################################
# 1. Start all policy servers in parallel
########################################
echo "========== Starting ${N} policy server(s) =========="

for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    PORT=${PORTS[$i]}
    POLICY_CONFIG=${POLICY_CONFIGS[$i]}
    POLICY_DIR=${POLICY_DIRS[$i]}

    if [ "$POLICY_MODE" = "custom" ]; then
        EXP_NAME="$(basename "$(dirname "$POLICY_DIR")")"
        CKPT_NUM="$(basename "$POLICY_DIR")"
    else
        EXP_NAME="default"
        CKPT_NUM="default"
    fi

    # One LOG_DIR per (config, suite, task-set) — shared across all TASK_IDS in
    # this run. The server partitions attn_vis/ into task<NN>_<name>/ep<NNN>/...
    # subfolders using the rollout context sent by main_vis.py, and replay
    # videos are named vis_task<NN>_ep<NNN>_*.mp4, so collisions are impossible.
    LOG_DIR="logs_vis/${EXP_NAME}/${CKPT_NUM}/${SUITE}/task${TASKS_TAG}"
    ATTN_DIR="${LOG_DIR}/attn_vis"
    mkdir -p "$LOG_DIR" "$ATTN_DIR"

    LOG_DIRS[$i]="$LOG_DIR"
    ATTN_DIRS[$i]="$ATTN_DIR"
    EXP_NAMES[$i]="$EXP_NAME"

    echo "  [${i}] GPU ${GPU} port ${PORT}  exp=${EXP_NAME}"

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
    PI0_ATTN_VIS_SKIP_CAMERAS="${ATTN_SKIP_CAMERAS}" \
    uv run scripts/serve_policy.py \
        --env LIBERO --port "${PORT}" \
        "${SERVER_CMD_EXTRA[@]}" \
        > "${LOG_DIR}/server.log" 2>&1 &
    SERVER_PIDS+=($!)
done

echo "Waiting for all servers to be ready..."
for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    SERVER_PID=${SERVER_PIDS[$i]}
    LOG_DIR=${LOG_DIRS[$i]}
    for attempt in $(seq 1 240); do
        if curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1; then
            echo "  [${i}] port ${PORT} ready."
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server [${i}] (port ${PORT}) died. See ${LOG_DIR}/server.log"
            exit 1
        fi
        if [ "$attempt" -eq 240 ]; then
            echo "ERROR: server [${i}] (port ${PORT}) did not start within 240s. See ${LOG_DIR}/server.log"
            exit 1
        fi
        sleep 1
    done
done

########################################
# 2. Start all eval clients in parallel
########################################
echo "========== Launching ${N} eval client(s) =========="

for i in "${!GPUS[@]}"; do
    PORT=${PORTS[$i]}
    LOG_DIR=${LOG_DIRS[$i]}
    EXP_NAME=${EXP_NAMES[$i]}
    echo "  [${i}] eval -> ${EXP_NAME}  (port ${PORT})  tasks=(${TASK_IDS[*]})"
    (
        source "$LIBERO_VENV"
        export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$LIBERO_THIRD_PARTY"
        python examples/libero/main_vis.py \
            --args.task-suite-name "${SUITE}" \
            --args.task-ids "${TASK_IDS[@]}" \
            --args.num-trials-per-task "${NUM_TRIALS}" \
            --args.port "${PORT}" \
            --args.video-out-path "${LOG_DIR}/videos"
    ) > "${LOG_DIR}/eval.log" 2>&1 &
    EVAL_PIDS+=($!)
done

echo "All ${N} jobs running."
echo "  Logs : logs_vis/<EXP_NAME>/<CKPT>/${SUITE}/task${TASKS_TAG}/"
echo "         (server.log, eval.log, videos/, attn_vis/task<NN>_<name>/ep<NNN>/...)"

########################################
# 3. Wait for evaluations to finish
########################################
FAIL=0
for i in "${!EVAL_PIDS[@]}"; do
    if wait "${EVAL_PIDS[$i]}"; then
        echo "  [DONE] ${EXP_NAMES[$i]}"
    else
        echo "  [FAIL] ${EXP_NAMES[$i]} (exit $?). See ${LOG_DIRS[$i]}/eval.log"
        FAIL=1
    fi
done

if [ "$FAIL" -eq 0 ]; then
    echo "========== All ${N} evaluations completed successfully =========="
else
    echo "========== Some evaluations failed. Check logs. =========="
    exit 1
fi
