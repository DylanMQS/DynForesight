#!/bin/bash
set -e

export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

#############################
# Configuration
#############################
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# Server policy: "default" uses Pi0_fast_libero weights; "custom" uses the checkpoint below.
POLICY_MODE="custom"  # "default" or "custom"


# GPUS=(0 1 2 3)
# PORTS=(8000 8001 8002 8003)
# POLICY_CONFIG="pi0_libero"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/pi0_libero_original_4gpu/5000"


# GPUS=(4 5 6 7)
# PORTS=(8004 8005 8006 8007)
# POLICY_CONFIG="pi0_libero"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/pi0_libero_original_4gpu/10000"


# GPUS=(0 1 2 3)
# PORTS=(8000 8001 8002 8003)
# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_cosmosdit0-8layer21_timestep200_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"



# GPUS=(4 5 6 7)
# PORTS=(8004 8005 8006 8007)
# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_cosmosdit0-8layer19_timestep200_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"





# GPUS=(0 1 2 3)
# PORTS=(8000 8001 8002 8003)
# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_wan21dit0-8layer3_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"


# GPUS=(0 1 2 3)
# PORTS=(8000 8001 8002 8003)
# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_wanvae0-8_only_temporal_contrastive_concat_weight015"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"


# GPUS=(0 1 2 3)
# PORTS=(8000 8001 8002 8003)
# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_wandit0-8layer21_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"



GPUS=(4 5 6 7)
PORTS=(8004 8005 8006 8007)
POLICY_CONFIG="pi0_libero_video_layer11_5e-1_wandit0-8layer25_multi_frame_concat"
POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"



# GPUS=(0 1 2 3)
# PORTS=(8000 8001 8002 8003)
# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_wandit0-8layer3_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"



# GPUS=(4 5 6 7)
# PORTS=(8004 8005 8006 8007)
# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_wandit0-8layer5_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"




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
LOG_DIR="logs/${EXP_NAME}/${CKPT_NUM}"

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
    for attempt in $(seq 1 120); do
        if curl -sf "http://localhost:${port}/healthz" >/dev/null 2>&1; then
            echo "  Server on port ${port} is ready."
            break
        fi
        if [ "$attempt" -eq 120 ]; then
            echo "ERROR: Server on port ${port} did not start within 120s. Check ${LOG_DIR}/server_${port}.log"
            exit 1
        fi
        sleep 1
    done
done

########################################
# 2. Start evaluations (libero env)
########################################
echo "========== Starting ${#SUITES[@]} evaluations (libero venv) =========="

for i in "${!SUITES[@]}"; do
    echo "  ${SUITES[$i]} -> localhost:${PORTS[$i]}"
    (
        source "$LIBERO_VENV"
        export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$LIBERO_THIRD_PARTY"
        python examples/libero/main.py \
            --args.task-suite-name "${SUITES[$i]}" \
            --args.port "${PORTS[$i]}" \
            --args.video-out-path "data/libero/videos/${SUITES[$i]}"
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
    echo "========== All evaluations completed successfully =========="
else
    echo "========== Some evaluations failed. Check logs. =========="
    exit 1
fi
