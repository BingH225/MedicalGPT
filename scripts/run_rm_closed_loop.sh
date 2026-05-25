#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-smoke}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
DATA_ROOT="${DATA_ROOT:-./data/medical_closed_loop/${PROFILE}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs-closed-loop-rm-qwen35-2b-${PROFILE}}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT="${GPU_COUNT:-1}"

if [[ "${GPU_COUNT}" -gt 1 ]]; then
  echo "Reward modeling currently stays on a single GPU in this repository. Falling back to GPU_COUNT=1." >&2
fi

TRAIN_DIR="${TRAIN_DIR:-${DATA_ROOT}/reward/train}"
VAL_DIR="${VAL_DIR:-${DATA_ROOT}/reward/val}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-64}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python3 training/reward_modeling.py \
  --model_name_or_path "${MODEL_NAME}" \
  --train_file_dir "${TRAIN_DIR}" \
  --validation_file_dir "${VAL_DIR}" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 2 \
  --do_train \
  --use_peft True \
  --seed 42 \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_steps 5 \
  --weight_decay 0.001 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 25 \
  --eval_strategy steps \
  --save_steps 100 \
  --save_strategy steps \
  --save_total_limit 3 \
  --max_source_length 1024 \
  --max_target_length 512 \
  --output_dir "${OUTPUT_DIR}" \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --fp16 \
  --torch_dtype float16 \
  --report_to tensorboard \
  --ddp_find_unused_parameters False \
  --remove_unused_columns False \
  --gradient_checkpointing True
