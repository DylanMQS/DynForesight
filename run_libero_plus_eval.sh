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

# GPUs to use. One policy server is started per GPU. Add/remove entries to scale.
GPUS=(0 1 2 3 4 5 6 7)
PORTS=(8000 8001 8002 8003 8004 8005 8006 8007)


# POLICY_CONFIG="pi05_libero_video_align_layer11_1e-1_wanvae0-8_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/pi05_libero_video_align_layer11_1e-1_wanvae0-8_multi_frame_concat_4gpu/30000"

POLICY_CONFIG="pi05_libero_video_align_layer11_1e-2_wanvae0-8_multi_frame_concat"
POLICY_DIR="checkpoints/${POLICY_CONFIG}/pi05_libero_video_align_layer11_1e-2_wanvae0-8_multi_frame_concat_4gpu/30000"


# POLICY_CONFIG="pi05_libero_video_align_layer11_1e-3_wanvae0-8_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"


# POLICY_CONFIG="pi05_libero"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/pi05_libero_original_4gpu/30000"

# POLICY_CONFIG="pi0_libero_video_layer11_5e-1_wanvae0-8_multi_frame_concat"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/${POLICY_CONFIG}_4gpu/30000"

# POLICY_CONFIG="pi0_libero"
# POLICY_DIR="checkpoints/${POLICY_CONFIG}/pi0_libero_original_4gpu/30000"


# Sharding granularity: each shard covers up to SHARD_SIZE consecutive tasks.
# 100 is the standard granularity used for this benchmark; do NOT go smaller
# without good reason (each shard pays a fixed Python/import startup cost).
# A suite with N tasks produces ceil(N / SHARD_SIZE) shards; the last one may
# be partial. With N~2500 and SHARD_SIZE=100 this gives ~26 shards/suite,
# ~104 shards total over 4 suites, scheduled across all GPUs.
SHARD_SIZE="${SHARD_SIZE:-100}"

# How many client processes are allowed to talk to the SAME server concurrently.
# Server inference is GPU-serialized so values >1 mainly help when the env
# simulator (CPU-bound) becomes the bottleneck. Keep at 1 for the safest setup.
CLIENTS_PER_GPU="${CLIENTS_PER_GPU:-1}"

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"

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
RESULTS_DIR="data/libero_plus/results/${EXP_NAME}/${CKPT_NUM}"
VIDEO_BASE_DIR="data/libero_plus/videos/${EXP_NAME}/${CKPT_NUM}"

if [ "${#GPUS[@]}" -ne "${#PORTS[@]}" ]; then
    echo "ERROR: GPUS (${#GPUS[@]}) and PORTS (${#PORTS[@]}) must have the same length."
    exit 1
fi

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

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$VIDEO_BASE_DIR"

echo "============================================================"
echo "  Experiment : ${EXP_NAME}/${CKPT_NUM}"
echo "  GPUs       : ${GPUS[*]}"
echo "  Ports      : ${PORTS[*]}"
echo "  Suites     : ${SUITES[*]}"
echo "  Chunks/su. : ${CHUNKS_PER_SUITE}  (total clients = $((${#SUITES[@]} * CHUNKS_PER_SUITE)))"
echo "  Clients/GP : ${CLIENTS_PER_GPU}"
echo "  Trials/task: ${NUM_TRIALS_PER_TASK}"
echo "  Log dir    : ${LOG_DIR}"
echo "  Results    : ${RESULTS_DIR}"
echo "============================================================"

########################################
# 1. Start one policy server per GPU
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
    for attempt in $(seq 1 600); do
        if curl -sf "http://localhost:${port}/healthz" >/dev/null 2>&1; then
            echo "  Server on port ${port} is ready."
            break
        fi
        if [ "$attempt" -eq 600 ]; then
            echo "ERROR: Server on port ${port} did not start within 600s. Check ${LOG_DIR}/server_${port}.log"
            exit 1
        fi
        sleep 1
    done
done

########################################
# 2. Build the (suite, start, end) shard list
########################################
# Need task counts per suite. Pull from task_classification.json which mirrors
# the python benchmark's n_tasks but doesn't require activating the env here.
declare -A SUITE_NTASKS
while IFS='=' read -r suite count; do
    SUITE_NTASKS["$suite"]="$count"
done < <(python3 - "$LIBERO_PLUS_DIR" "${SUITES[@]}" <<'PY'
import json, sys
libero_root = sys.argv[1]
suites = sys.argv[2:]
with open(f"{libero_root}/libero/libero/benchmark/task_classification.json") as f:
    cls = json.load(f)
for s in suites:
    print(f"{s}={len(cls[s])}")
PY
)

SHARD_SUITE=()
SHARD_START=()
SHARD_END=()

if [ -n "${SHARDS_FILE:-}" ]; then
    # Resume / partial mode: read explicit shard list from a file.
    # Format: one shard per line, "<suite> <start> <end>", '#' starts a comment.
    if [ ! -f "$SHARDS_FILE" ]; then
        echo "ERROR: SHARDS_FILE=$SHARDS_FILE not found"
        exit 1
    fi
    echo "Reading shard list from $SHARDS_FILE (overriding auto-sharding)"
    while read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        [ -z "${line// /}" ] && continue
        # shellcheck disable=SC2206
        parts=($line)
        if [ "${#parts[@]}" -ne 3 ]; then
            echo "ERROR: bad line in SHARDS_FILE (expected 'suite start end'): $line"
            exit 1
        fi
        SHARD_SUITE+=("${parts[0]}")
        SHARD_START+=("${parts[1]}")
        SHARD_END+=("${parts[2]}")
    done < "$SHARDS_FILE"
else
    for suite in "${SUITES[@]}"; do
        n=${SUITE_NTASKS[$suite]:-0}
        if [ "$n" -le 0 ]; then
            echo "ERROR: could not resolve task count for suite=${suite}"
            exit 1
        fi
        cur=0
        while [ "$cur" -lt "$n" ]; do
            nxt=$((cur + SHARD_SIZE))
            if [ "$nxt" -gt "$n" ]; then nxt=$n; fi
            SHARD_SUITE+=("$suite")
            SHARD_START+=("$cur")
            SHARD_END+=("$nxt")
            cur=$nxt
        done
    done
fi

NUM_SHARDS=${#SHARD_SUITE[@]}
if [ "$NUM_SHARDS" -le 0 ]; then
    echo "Nothing to run (NUM_SHARDS=0). Exiting."
    exit 0
fi
echo "========== Built ${NUM_SHARDS} shards across ${#SUITES[@]} suites =========="

########################################
# 3. Launch eval clients with round-robin GPU assignment +
#    per-GPU concurrency cap (CLIENTS_PER_GPU)
########################################
NUM_GPUS=${#GPUS[@]}
SLOTS=$((NUM_GPUS * CLIENTS_PER_GPU))

# Per-GPU running-client counters and per-shard metadata.
declare -a GPU_INFLIGHT
for ((g=0; g<NUM_GPUS; g++)); do GPU_INFLIGHT[$g]=0; done
declare -a EVAL_GPU_IDX
declare -a EVAL_SUITE
declare -a EVAL_START
declare -a EVAL_END

launch_shard() {
    local idx=$1
    local gpu_idx=$2
    local suite=${SHARD_SUITE[$idx]}
    local s=${SHARD_START[$idx]}
    local e=${SHARD_END[$idx]}
    local port=${PORTS[$gpu_idx]}
    local gpu=${GPUS[$gpu_idx]}
    local log="${LOG_DIR}/eval_${suite}_${s}_${e}.log"
    echo "  [shard $((idx+1))/${NUM_SHARDS}] suite=${suite} range=[${s},${e}) -> GPU ${gpu} port ${port}"
    (
        source "$LIBERO_PLUS_VENV"
        export LIBERO_CONFIG_PATH="$HOME/.libero_plus"
        export LIBERO_PLUS_DIR="$LIBERO_PLUS_DIR"
        export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$LIBERO_PLUS_DIR"
        export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
        python examples/libero/main_plus.py \
            --args.task-suite-name "$suite" \
            --args.port "$port" \
            --args.num-trials-per-task "$NUM_TRIALS_PER_TASK" \
            --args.start-idx "$s" \
            --args.end-idx "$e" \
            --args.video-out-path "${VIDEO_BASE_DIR}/${suite}" \
            --args.results-out-dir "${RESULTS_DIR}"
    ) > "$log" 2>&1 &
    local pid=$!
    EVAL_PIDS+=("$pid")
    EVAL_GPU_IDX+=("$gpu_idx")
    EVAL_SUITE+=("$suite")
    EVAL_START+=("$s")
    EVAL_END+=("$e")
    GPU_INFLIGHT[$gpu_idx]=$((GPU_INFLIGHT[gpu_idx] + 1))
}

# Pick the GPU with the lowest current in-flight count (ties: smallest index),
# subject to CLIENTS_PER_GPU.
pick_free_gpu() {
    local best=-1
    local best_load=$((CLIENTS_PER_GPU + 1))
    for ((g=0; g<NUM_GPUS; g++)); do
        local load=${GPU_INFLIGHT[$g]}
        if [ "$load" -lt "$CLIENTS_PER_GPU" ] && [ "$load" -lt "$best_load" ]; then
            best=$g
            best_load=$load
        fi
    done
    echo "$best"
}

# Reap any finished EVAL_PIDS, decrement GPU counters, return number reaped.
reap_finished() {
    local reaped=0
    local i
    for i in "${!EVAL_PIDS[@]}"; do
        local pid=${EVAL_PIDS[$i]}
        # already reaped sentinel
        if [ -z "$pid" ]; then continue; fi
        if ! kill -0 "$pid" 2>/dev/null; then
            local rc=0
            wait "$pid" 2>/dev/null || rc=$?
            local g=${EVAL_GPU_IDX[$i]}
            GPU_INFLIGHT[$g]=$((GPU_INFLIGHT[g] - 1))
            if [ "$rc" -eq 0 ]; then
                echo "    [DONE] ${EVAL_SUITE[$i]} [${EVAL_START[$i]},${EVAL_END[$i]}) (GPU ${GPUS[$g]})"
            else
                echo "    [FAIL] ${EVAL_SUITE[$i]} [${EVAL_START[$i]},${EVAL_END[$i]}) (GPU ${GPUS[$g]}) rc=${rc}"
                FAIL=1
            fi
            EVAL_PIDS[$i]=""
            reaped=$((reaped + 1))
        fi
    done
    return $reaped
}

FAIL=0
echo "========== Launching ${NUM_SHARDS} eval clients (max ${SLOTS} concurrent) =========="
shard_idx=0
while [ "$shard_idx" -lt "$NUM_SHARDS" ]; do
    gpu_idx=$(pick_free_gpu)
    if [ "$gpu_idx" -lt 0 ]; then
        # All slots busy - wait briefly and reap finished children.
        sleep 2
        reap_finished || true
        continue
    fi
    launch_shard "$shard_idx" "$gpu_idx"
    shard_idx=$((shard_idx + 1))
done

echo "All evaluations launched. Waiting for completion..."
echo "  Log dir     : ${LOG_DIR}/"
echo "  Server logs : ${LOG_DIR}/server_<port>.log"
echo "  Eval logs   : ${LOG_DIR}/eval_<suite>_<start>_<end>.log"
echo "  Per-shard   : ${RESULTS_DIR}/<suite>/<start>_<end>.json"

########################################
# 4. Wait for any remaining clients
########################################
while :; do
    still_running=0
    for i in "${!EVAL_PIDS[@]}"; do
        pid=${EVAL_PIDS[$i]}
        if [ -z "$pid" ]; then continue; fi
        if kill -0 "$pid" 2>/dev/null; then
            still_running=1
        fi
    done
    if [ "$still_running" -eq 0 ]; then
        reap_finished || true
        break
    fi
    reap_finished || true
    sleep 2
done

########################################
# 5. Aggregate per-shard json into overall_results.json
########################################
echo "========== Aggregating per-shard results =========="
AGG_OUT="${RESULTS_DIR}/overall_results.json"
python3 examples/libero/aggregate_libero_plus.py \
    --results_dir "$RESULTS_DIR" \
    --suites "${SUITES[@]}" \
    --output "$AGG_OUT" \
    | tee "${LOG_DIR}/aggregate.log"
echo "  Wrote ${AGG_OUT}"

if [ "$FAIL" -eq 0 ]; then
    echo "========== All LIBERO-Plus evaluations completed successfully =========="
else
    echo "========== Some shards failed. Check logs in ${LOG_DIR}/ =========="
    exit 1
fi

exit 0
}
