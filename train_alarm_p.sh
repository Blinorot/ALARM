#!/usr/bin/env bash
set -euo pipefail

### TRAINING SCRIPTS TO REPRODUCE ALARM-P
# First, train AL-Whisper-Instruct-R.
# Then, extract its pretrained weights.
# Finally, load them into ALARM-P and train the fusion module.

# Single-Encoder: AL-Whisper-Instruct-R
# Runs on 2 GPUs. Can be run on 40 GB.
# ~24-32h.
accelerate launch \
    --config-file src/configs/accelerate/multigpu.yaml \
    --num-processes 2 \
    train.py \
    model=whisper_batch \
    dataset=content_4b \
    training_args.per_device_train_batch_size=2 \
    training_args.per_device_eval_batch_size=2 \
    training_args.gradient_accumulation_steps=8 \
    training_args.num_train_epochs=2 \
    training_args.output_dir="saved/Qwen3-4B-Thinking-2507/content_whisper_batch" \
    training_args.eval_steps=5000 \
    training_args.dataloader_num_workers=2 \
    model.llm="Qwen/Qwen3-4B-Thinking-2507" \
    model.llm_embedding_dim=2560

# Extract pretrained weights.
python3 get_pretrained_adapters.py \
    --whisper "saved/Qwen3-4B-Thinking-2507/content_whisper_batch/checkpoint-34960" \
    --save-dir "data/pretrained_weights"

# Finally, train ALARM-P.
# The last four arguments ensure that the adapter weights are loaded and kept frozen.
# Since only Whisper is pretrained and frozen, we specify it in the last two arguments.
# Runs on 4 GPUs (H200). ~9-12 days.
accelerate launch \
    --config-file src/configs/accelerate/multigpu.yaml \
    --num-processes 4 \
    train.py \
    model=alarm_p \
    dataset=all_4b \
    training_args.per_device_train_batch_size=4 \
    training_args.per_device_eval_batch_size=4 \
    training_args.gradient_accumulation_steps=4 \
    training_args.num_train_epochs=2 \
    training_args.output_dir="saved/Qwen3-4B-Thinking-2507/alldataset_alarm_p" \
    training_args.eval_steps=20000 \
    training_args.dataloader_num_workers=2 \
    model.llm="Qwen/Qwen3-4B-Thinking-2507" \
    model.llm_embedding_dim=2560 \
    pretrained_adapters_dir=data/pretrained_weights \
    freeze_adapters=True \
    pretrained_adapters_names='["whisper"]' \
    frozen_adapters_names='["whisper"]'
