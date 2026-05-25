#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-smoke}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
DATA_ROOT="${DATA_ROOT:-./data/medical_closed_loop/${PROFILE}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs-closed-loop-dpo-qwen35-2b-${PROFILE}}"
GPU_COUNT="${GPU_COUNT:-1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"

TRAIN_DIR="${TRAIN_DIR:-${DATA_ROOT}/reward/train}"
VAL_DIR="${VAL_DIR:-${DATA_ROOT}/reward/val}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-64}"

CMD=(
  training/dpo_training.py
  --model_name_or_path "${MODEL_NAME}"
  --train_file_dir "${TRAIN_DIR}"
  --validation_file_dir "${VAL_DIR}"
  --per_device_train_batch_size 2
  --gradient_accumulation_steps 8
  --per_device_eval_batch_size 2
  --do_train
  --do_eval
  --use_peft True
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --max_eval_samples "${MAX_EVAL_SAMPLES}"
  --max_steps 100
  --eval_steps 20
  --save_steps 50
  --max_source_length 1024
  --max_target_length 512
  --output_dir "${OUTPUT_DIR}"
  --target_modules all
  --lora_rank 8
  --lora_alpha 16
  --lora_dropout 0.05
  --torch_dtype float16
  --fp16 True
  --bf16 False
  --report_to tensorboard
  --remove_unused_columns False
  --gradient_checkpointing True
  --cache_dir ./cache
)

if [[ "${GPU_COUNT}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" torchrun --nproc_per_node "${GPU_COUNT}" "${CMD[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python3 "${CMD[@]}"
fi
