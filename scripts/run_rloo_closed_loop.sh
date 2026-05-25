#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-smoke}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${MODEL_NAME}}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_NAME}}"
DATA_ROOT="${DATA_ROOT:-./data/medical_closed_loop/${PROFILE}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs-closed-loop-rloo-qwen35-2b-${PROFILE}}"
GPU_COUNT="${GPU_COUNT:-1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"

TRAIN_DIR="${TRAIN_DIR:-${DATA_ROOT}/sft/train}"
VAL_DIR="${VAL_DIR:-${DATA_ROOT}/sft/val}"
MAX_STEPS="${MAX_STEPS:-80}"

CMD=(
  training/ppo_training.py
  --sft_model_path "${SFT_MODEL_PATH}"
  --reward_model_path "${REWARD_MODEL_PATH}"
  --model_name_or_path "${MODEL_NAME}"
  --dtype float16
  --train_file_dir "${TRAIN_DIR}"
  --validation_file_dir "${VAL_DIR}"
  --max_source_length 1024
  --max_completion_length 768
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 4
  --gradient_checkpointing True
  --do_train
  --max_steps "${MAX_STEPS}"
  --output_dir "${OUTPUT_DIR}"
  --eval_strategy steps
  --eval_steps 20
  --num_train_epochs 1
  --report_to tensorboard
  --dataset_num_proc 1
)

if [[ "${GPU_COUNT}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" torchrun --nproc_per_node "${GPU_COUNT}" "${CMD[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python3 "${CMD[@]}"
fi
