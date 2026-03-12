#!/usr/bin/env bash
set -euo pipefail

### TRAINING SCRIPTS TO REPRODUCE ALARM-CA
# First, train the single-encoder models.
# Then, extract their pretrained weights.
# Finally, load them into ALARM-CA and train the fusion module.

# Single-Encoder: AL-MuQ-R, AL-W2VBERT2-R, AL-SSLAM-R
# Runs on 2 GPUs. W2VBERT and SSLAM can be run on 40 GB, MuQ on 80 GB.
# ~24-32h each.
models=("muq" "w2vbert" "sslam")
datasets=("music_4b" "speech_4b" "audio_4b")
output_dirs=(
  "saved/Qwen3-4B-Thinking-2507/music_muq"
  "saved/Qwen3-4B-Thinking-2507/speech_w2vbert_layer"
  "saved/Qwen3-4B-Thinking-2507/audio_sslam_layer"
)

for i in "${!models[@]}"; do
  accelerate launch \
    --config-file src/configs/accelerate/multigpu.yaml \
    --num-processes 2 \
    train.py \
    model="${models[$i]}" \
    dataset="${datasets[$i]}" \
    training_args.per_device_train_batch_size=2 \
    training_args.per_device_eval_batch_size=2 \
    training_args.gradient_accumulation_steps=8 \
    training_args.num_train_epochs=2 \
    training_args.output_dir="${output_dirs[$i]}" \
    training_args.eval_steps=5000 \
    training_args.dataloader_num_workers=2 \
    model.llm="Qwen/Qwen3-4B-Thinking-2507" \
    model.llm_embedding_dim=2560
done

# Single-Encoder: AL-Whisper-R (All data)
# Runs on 4GPUs. We use H200. ~6-9 days
# For faster training, you can change eval_steps and
# replace all_4b with all_4b_faster_eval to skip some evaluations.
# Optionally you can add sample_generations=false
accelerate launch \
    --config-file src/configs/accelerate/multigpu.yaml \
    --num-processes 4 \
    train.py \
    model=whisper_batch \
    dataset=all_4b \
    training_args.per_device_train_batch_size=4 \
    training_args.per_device_eval_batch_size=4 \
    training_args.gradient_accumulation_steps=4 \
    training_args.num_train_epochs=2 \
    training_args.output_dir="saved/Qwen3-4B-Thinking-2507/alldataset_whisper_batch" \
    training_args.eval_steps=20000 \
    training_args.dataloader_num_workers=2 \
    model.llm="Qwen/Qwen3-4B-Thinking-2507" \
    model.llm_embedding_dim=2560

# Now you need to extract pre-trained weights
python3 get_pretrained_adapters.py \
    --whisper "saved/Qwen3-4B-Thinking-2507/alldataset_whisper_batch/checkpoint-172380" \
    --w2vbert "saved/Qwen3-4B-Thinking-2507/speech_w2vbert_layer/checkpoint-31376" \
    --sslam "saved/Qwen3-4B-Thinking-2507/audio_sslam_layer/checkpoint-27010" \
    --muq "saved/Qwen3-4B-Thinking-2507/music_muq/checkpoint-25522" \
    --save-dir "data/pretrained_weights"

# Finally, train ALARM-CA
# Last two arguments ensure that the adapter weights will be loaded and kept frozen
# 4 GPUs (H200). ~9-12 days.
accelerate launch \
    --config-file src/configs/accelerate/multigpu.yaml \
    --num-processes 4 \
    train.py \
    model=alarm_ca \
    dataset=all_4b \
    training_args.per_device_train_batch_size=4 \
    training_args.per_device_eval_batch_size=4 \
    training_args.gradient_accumulation_steps=4 \
    training_args.num_train_epochs=2 \
    training_args.output_dir="saved/Qwen3-4B-Thinking-2507/alldataset_alarm_ca" \
    training_args.eval_steps=20000 \
    training_args.dataloader_num_workers=2 \
    model.llm="Qwen/Qwen3-4B-Thinking-2507" \
    model.llm_embedding_dim=2560 \
    pretrained_adapters_dir=data/pretrained_weights \
    freeze_adapters=True
