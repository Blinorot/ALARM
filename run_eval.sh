#!/usr/bin/env bash

DATASET="$1"
MODEL="$2"
CHECKPOINT="$3"
DATASET_SPLIT="${4:-null}"
BATCH_SIZE="${5:-8}"
GPU_MEMORY_UTILIZATION="${6:-0.5}"
SEED="${7:-null}"

echo "Running evaluation script with the following arguments:
    DATASET=$DATASET
    MODEL=$MODEL
    CHECKPOINT=$CHECKPOINT
    DATASET_SPLIT=$DATASET_SPLIT
    BATCH_SIZE=$BATCH_SIZE
    GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    SEED=$SEED
"

for max_thinking_tokens in 8192; do
    python3  generate_vllm_mcqa.py model=$MODEL \
        checkpoint_name=$CHECKPOINT \
        cuda_devices="'$CUDA_VISIBLE_DEVICES'" \
        dataset.name=$DATASET \
        dataset.split=$DATASET_SPLIT \
        model.llm="Qwen/Qwen3-4B-Thinking-2507" \
        model.llm_embedding_dim=2560 \
        batch_size=$BATCH_SIZE \
        seed=$SEED \
        model.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
        model.max_tokens=16384 \
        model.max_thinking_tokens=$max_thinking_tokens
done
