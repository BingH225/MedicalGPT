#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

ensure_common_dirs
"${SCRIPT_DIR}/prepare_stage_data.sh"
"${SCRIPT_DIR}/bootstrap_remote.sh"

submission_stamp="$(date +%Y%m%d_%H%M%S)"
submission_log="${OUTPUT_ROOT}/pipeline_submission_${submission_stamp}.txt"

sft_job_id="$(qsub "${SCRIPT_DIR}/run_sft_qwen35_2b.pbs")"
rm_job_id="$(qsub -W depend=afterok:${sft_job_id} "${SCRIPT_DIR}/run_rm_qwen35_2b.pbs")"
rloo_job_id="$(qsub -W depend=afterok:${sft_job_id}:${rm_job_id} "${SCRIPT_DIR}/run_rloo_qwen35_2b.pbs")"
dpo_job_id="$(qsub -W depend=afterok:${sft_job_id} "${SCRIPT_DIR}/run_dpo_qwen35_2b.pbs")"

{
  echo "submission_time=$(date -Iseconds)"
  echo "sft_job_id=${sft_job_id}"
  echo "rm_job_id=${rm_job_id}"
  echo "rloo_job_id=${rloo_job_id}"
  echo "dpo_job_id=${dpo_job_id}"
  echo "pbs_log_root=${PBS_LOG_ROOT}"
  echo "artifact_root=${ARTIFACT_ROOT}"
} | tee "${submission_log}"
