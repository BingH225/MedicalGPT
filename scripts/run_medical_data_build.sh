#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./data/medical_closed_loop/${PROFILE}}"
LOCAL_SFT_SOURCE="${LOCAL_SFT_SOURCE:-./data/sft/medical_sft_1K_format.jsonl}"
GENERAL_QA_RATIO="${GENERAL_QA_RATIO:-0.2}"
INCLUDE_HF_PUBLIC_SOURCES="${INCLUDE_HF_PUBLIC_SOURCES:-0}"
HF_MAX_RECORDS="${HF_MAX_RECORDS:--1}"
SFT_TARGET="${SFT_TARGET:-0}"
PREFERENCE_TARGET="${PREFERENCE_TARGET:-0}"
EVAL_TARGET="${EVAL_TARGET:-0}"

ARGS=(
  build
  --profile "${PROFILE}"
  --output-root "${OUTPUT_ROOT}"
  --local-sft-source "${LOCAL_SFT_SOURCE}"
  --general-qa-ratio "${GENERAL_QA_RATIO}"
  --hf-max-records "${HF_MAX_RECORDS}"
)

if [[ "${INCLUDE_HF_PUBLIC_SOURCES}" == "1" ]]; then
  ARGS+=(--include-hf-public-sources)
fi

if [[ "${SFT_TARGET}" != "0" ]]; then
  ARGS+=(--sft-target "${SFT_TARGET}")
fi
if [[ "${PREFERENCE_TARGET}" != "0" ]]; then
  ARGS+=(--preference-target "${PREFERENCE_TARGET}")
fi
if [[ "${EVAL_TARGET}" != "0" ]]; then
  ARGS+=(--eval-target "${EVAL_TARGET}")
fi

python tools/medical_closed_loop.py "${ARGS[@]}"
