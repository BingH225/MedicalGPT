#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-smoke}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-2B}"
DATA_ROOT="${DATA_ROOT:-./data/medical_closed_loop/${PROFILE}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs-closed-loop-sft-qwen35-2b-${PROFILE}}"
GPU_COUNT="${GPU_COUNT:-1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"

TRAIN_DIR="${TRAIN_DIR:-${DATA_ROOT}/sft/train}"
VAL_DIR="${VAL_DIR:-${DATA_ROOT}/sft/val}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-64}"
GRAD_ACC="${GRAD_ACC:-8}"
TRAIN_BATCH="${TRAIN_BATCH:-2}"
EVAL_BATCH="${EVAL_BATCH:-2}"

CMD=(
  training/supervised_finetuning.py
  --model_name_or_path "${MODEL_NAME}"
  --train_file_dir "${TRAIN_DIR}"
  --validation_file_dir "${VAL_DIR}"
  --per_device_train_batch_size "${TRAIN_BATCH}"
  --per_device_eval_batch_size "${EVAL_BATCH}"
  --do_train
  --do_eval
  --use_peft True
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --max_eval_samples "${MAX_EVAL_SAMPLES}"
  --model_max_length 1024
  --num_train_epochs 1
  --learning_rate 2e-5
  --warmup_steps 5
  --weight_decay 0.05
  --logging_strategy steps
  --logging_steps 10
  --eval_steps 25
  --eval_strategy steps
  --save_steps 100
  --save_strategy steps
  --save_total_limit 3
  --gradient_accumulation_steps "${GRAD_ACC}"
  --preprocessing_num_workers 4
  --output_dir "${OUTPUT_DIR}"
  --ddp_timeout 30000
  --logging_first_step True
  --target_modules all
  --lora_rank 8
  --lora_alpha 16
  --lora_dropout 0.05
  --torch_dtype float16
  --fp16
  --report_to tensorboard
  --ddp_find_unused_parameters False
  --gradient_checkpointing True
  --cache_dir ./cache
)

if [[ "${GPU_COUNT}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" torchrun --nproc_per_node "${GPU_COUNT}" "${CMD[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python3 "${CMD[@]}"
fi
