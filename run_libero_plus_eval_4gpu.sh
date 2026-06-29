#!/bin/bash
# NOTE: Wrapped in `{ ... }` so bash parses the whole script before executing
# anything. Otherwise editing this file *while it is running* (bash reads it
# line by line at runtime) can cause spurious "unexpected EOF" errors
# at fictional line numbers near the end.
{
set -e

export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

#############################
# Configuration
#############################
# Order matters: shards are enqueued suite-by-suite in this order, and the
# scheduler picks them in queue order. Put the longest-per-episode suite first
# (LPT-style) so the long tail starts early and doesn't drag out the makespan.
# max_steps: libero_10=520, libero_goal=300, libero_object=280, libero_spatial=220
SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")

# Server policy: "default" uses Pi0_fast_libero weights; "custom" uses the checkpoint below.
POLICY_MODE="custom"  # "default" or "custom"


GPUS=(0 1 2 3)
PORTS=(8000 8001 8002 8003)
POLICY_CONFIG="pi0_libero"
POLICY_DIR="checkpoints/${POLICY_CONFIG}/pi0_libero_original_4gpu/30000"


# GPUS=(4 5 6 7)
# PORTS=(8004 8005 8006 8007)
# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_wandit0-8layer9_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"




#############################
# Derived paths
#############################
LIBERO_PLUS_VENV="$SCRIPT_DIR/examples/libero/.venv_plus/bin/activate"
LIBERO_PLUS_DIR="/mnt/data/mqs/workspace/VLA/LIBERO-plus"

if [ "$POLICY_MODE" = "custom" ]; then
    EXP_NAME="$(basename "$(dirname "$POLICY_DIR")")"
    CKPT_NUM="$(basename "$POLICY_DIR")"
else
    EXP_NAME="default"
    CKPT_NUM="default"
fi
LOG_DIR="logs_plus/${EXP_NAME}/${CKPT_NUM}"

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

mkdir -p "$LOG_DIR"

########################################
# 1. Start policy servers (pi05 env via uv)
########################################
echo "========== Starting ${#GPUS[@]} policy servers (uv run) =========="

SERVER_CMD_EXTRA=()
if [ "$POLICY_MODE" = "custom" ]; then
    SERVER_CMD_EXTRA=(policy:checkpoint --policy.config "$POLICY_CONFIG" --policy.dir "$POLICY_DIR")
fi

for i in "${!GPUS[@]}"; do
    echo "  GPU ${GPUS[$i]} | port ${PORTS[$i]}"
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} \
    TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache_gpu${GPUS[$i]}" \
    uv run scripts/serve_policy.py \
        --env LIBERO --port "${PORTS[$i]}" \
        "${SERVER_CMD_EXTRA[@]}" \
        > "${LOG_DIR}/server_${PORTS[$i]}.log" 2>&1 &
    SERVER_PIDS+=($!)
done

echo "Waiting for servers to be ready..."
for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    for attempt in $(seq 1 300); do
        if curl -sf "http://localhost:${port}/healthz" >/dev/null 2>&1; then
            echo "  Server on port ${port} is ready."
            break
        fi
        if [ "$attempt" -eq 300 ]; then
            echo "ERROR: Server on port ${port} did not start within 300s. Check ${LOG_DIR}/server_${port}.log"
            exit 1
        fi
        sleep 1
    done
done

########################################
# 2. Start evaluations (libero-plus env)
########################################
echo "========== Starting ${#SUITES[@]} LIBERO-Plus evaluations =========="

for i in "${!SUITES[@]}"; do
    echo "  ${SUITES[$i]} -> localhost:${PORTS[$i]}"
    (
        source "$LIBERO_PLUS_VENV"
        export LIBERO_CONFIG_PATH="$HOME/.libero_plus"
        export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$LIBERO_PLUS_DIR"
        export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
        python examples/libero/main.py \
            --args.task-suite-name "${SUITES[$i]}" \
            --args.port "${PORTS[$i]}" \
            --args.num-trials-per-task 1 \
            --args.video-out-path "data/libero_plus/videos/${SUITES[$i]}"
    ) > "${LOG_DIR}/eval_${SUITES[$i]}.log" 2>&1 &
    EVAL_PIDS+=($!)
done

echo "All evaluations launched. Waiting for completion..."
echo "  Log dir     : ${LOG_DIR}/"
echo "  Server logs : ${LOG_DIR}/server_<port>.log"
echo "  Eval logs   : ${LOG_DIR}/eval_<suite>.log"

########################################
# 3. Wait for evaluations to finish
########################################
FAIL=0
for i in "${!EVAL_PIDS[@]}"; do
    if wait "${EVAL_PIDS[$i]}"; then
        echo "  [DONE] ${SUITES[$i]}"
    else
        echo "  [FAIL] ${SUITES[$i]} (exit code $?)"
        FAIL=1
    fi
done

if [ "$FAIL" -eq 0 ]; then
    echo "========== All LIBERO-Plus evaluations completed successfully =========="
else
    echo "========== Some evaluations failed. Check logs. =========="
    exit 1
fi

exit 0
}
