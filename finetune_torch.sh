train_config_name=$1
model_name=$2
gpu_use=$3

export CUDA_VISIBLE_DEVICES=$gpu_use

# Count number of GPUs
IFS=',' read -ra GPU_ARRAY <<< "$gpu_use"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Using GPUs: $CUDA_VISIBLE_DEVICES (count: $NUM_GPUS)"

if [ "$NUM_GPUS" -gt 1 ]; then
    .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        scripts/train_pytorch.py $train_config_name --exp-name=$model_name --overwrite
else
    .venv/bin/python scripts/train_pytorch.py $train_config_name --exp-name=$model_name --overwrite
fi
