#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

ensure_common_dirs
"${SCRIPT_DIR}/prepare_stage_data.sh"
"${SCRIPT_DIR}/bootstrap_remote.sh"

submission_stamp="$(date +%Y%m%d_%H%M%S)"
submission_log="${OUTPUT_ROOT}/sft_submission_${submission_stamp}.txt"
sft_job_id="$(qsub "${SCRIPT_DIR}/run_sft_qwen35_2b.pbs")"

{
  echo "submission_time=$(date -Iseconds)"
  echo "sft_job_id=${sft_job_id}"
  echo "pbs_log_root=${PBS_LOG_ROOT}"
  echo "artifact_root=${ARTIFACT_ROOT}"
} | tee "${submission_log}"
