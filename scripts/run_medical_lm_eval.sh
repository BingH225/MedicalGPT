#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-smoke}"
DATA_ROOT="${DATA_ROOT:-./data/medical_closed_loop/${PROFILE}}"
EVAL_FILE="${EVAL_FILE:-${DATA_ROOT}/eval/medical_eval_cases.jsonl}"
TASK_NAME="${TASK_NAME:-medical_structured_case_${PROFILE}}"
TASK_FILE="${TASK_FILE:-./eval/lm_eval/generated/${TASK_NAME}.yaml}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-2B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results/medical_closed_loop/${TASK_NAME}}"
PREDICTION_FILE="${PREDICTION_FILE:-${OUTPUT_ROOT}/samples.jsonl}"

python tools/medical_lm_eval.py write-task \
  --eval-file "${EVAL_FILE}" \
  --output "${TASK_FILE}" \
  --task-name "${TASK_NAME}"

lm_eval run \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},trust_remote_code=True" \
  --tasks "${TASK_FILE}" \
  --output_path "${OUTPUT_ROOT}" \
  --log_samples

if [[ ! -f "${PREDICTION_FILE}" ]]; then
  DETECTED_FILE="$(find "${OUTPUT_ROOT}" -type f -name '*.jsonl' | head -n 1 || true)"
  if [[ -z "${DETECTED_FILE}" ]]; then
    echo "No lm-eval sample JSONL file was found under ${OUTPUT_ROOT}." >&2
    exit 1
  fi
  PREDICTION_FILE="${DETECTED_FILE}"
fi

python tools/medical_lm_eval.py score \
  --eval-file "${EVAL_FILE}" \
  --predictions "${PREDICTION_FILE}" \
  --output "${OUTPUT_ROOT}/structured_metrics.json" \
  --per-case-output "${OUTPUT_ROOT}/structured_metrics_per_case.jsonl"

python tools/medical_lm_eval.py export-rubric \
  --eval-file "${EVAL_FILE}" \
  --predictions "${PREDICTION_FILE}" \
  --output "${OUTPUT_ROOT}/manual_review.csv"
