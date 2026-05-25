#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

ensure_common_dirs

cp "${PROJECT_ROOT}/data/sft/medical_sft_1K_format.jsonl" "${HOPPER_SFT_DATA_DIR}/medical_sft_1K_format.jsonl"
cp "${PROJECT_ROOT}/data/reward/dpo_zh_500.jsonl" "${HOPPER_REWARD_DATA_DIR}/dpo_zh_500.jsonl"

echo "[prepare_stage_data] sft_dir=${HOPPER_SFT_DATA_DIR}"
echo "[prepare_stage_data] reward_dir=${HOPPER_REWARD_DATA_DIR}"
