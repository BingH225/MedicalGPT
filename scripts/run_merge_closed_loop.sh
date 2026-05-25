#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"
LORA_MODEL="${LORA_MODEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./artifacts/closed_loop_merged}"
RESIZE_EMB="${RESIZE_EMB:-0}"

if [[ -z "${LORA_MODEL}" ]]; then
  echo "LORA_MODEL is required." >&2
  exit 1
fi

ARGS=(
  --base_model "${BASE_MODEL}"
  --tokenizer_path "${TOKENIZER_PATH}"
  --lora_model "${LORA_MODEL}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${RESIZE_EMB}" == "1" ]]; then
  ARGS+=(--resize_emb)
fi

python tools/merge_peft_adapter.py "${ARGS[@]}"
