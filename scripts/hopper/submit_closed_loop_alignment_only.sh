#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

ensure_common_dirs
"${SCRIPT_DIR}/bootstrap_remote.sh"

submission_stamp="$(date +%Y%m%d_%H%M%S)"
submission_log="${OUTPUT_ROOT}/closed_loop_alignment_submission_${submission_stamp}.txt"

QSUB_ENV_VARS="MEDICAL_CLOSED_LOOP_PROFILE=${MEDICAL_CLOSED_LOOP_PROFILE},MEDICAL_CLOSED_LOOP_ROOT=${MEDICAL_CLOSED_LOOP_ROOT},MEDICAL_CLOSED_LOOP_EVAL_FILE=${MEDICAL_CLOSED_LOOP_EVAL_FILE},HOPPER_SFT_TRAIN_DIR=${MEDICAL_CLOSED_LOOP_ROOT}/sft/train,HOPPER_SFT_VAL_DIR=${MEDICAL_CLOSED_LOOP_ROOT}/sft/val,HOPPER_REWARD_TRAIN_DIR=${MEDICAL_CLOSED_LOOP_ROOT}/reward/train,HOPPER_REWARD_VAL_DIR=${MEDICAL_CLOSED_LOOP_ROOT}/reward/val,DATA_BUILD_INCLUDE_HF_PUBLIC_SOURCES=${DATA_BUILD_INCLUDE_HF_PUBLIC_SOURCES},DATA_BUILD_HF_MAX_RECORDS=${DATA_BUILD_HF_MAX_RECORDS},DATA_BUILD_SFT_TARGET=${DATA_BUILD_SFT_TARGET},DATA_BUILD_PREFERENCE_TARGET=${DATA_BUILD_PREFERENCE_TARGET},DATA_BUILD_EVAL_TARGET=${DATA_BUILD_EVAL_TARGET},DATA_BUILD_GENERAL_QA_RATIO=${DATA_BUILD_GENERAL_QA_RATIO},DATA_BUILD_MIN_VALIDATION_SIZE=${DATA_BUILD_MIN_VALIDATION_SIZE},DATA_BUILD_MIN_PREFERENCE_VALIDATION_SIZE=${DATA_BUILD_MIN_PREFERENCE_VALIDATION_SIZE},DATA_BUILD_LOCAL_SFT_SOURCE=${DATA_BUILD_LOCAL_SFT_SOURCE},BASE_MODEL=${BASE_MODEL},SFT_MAX_TRAIN_SAMPLES=${SFT_MAX_TRAIN_SAMPLES},RM_MAX_TRAIN_SAMPLES=${RM_MAX_TRAIN_SAMPLES},DPO_MAX_TRAIN_SAMPLES=${DPO_MAX_TRAIN_SAMPLES},RLOO_MAX_STEPS=${RLOO_MAX_STEPS}"

SFT_JOB_ID="${SFT_JOB_ID:-}"

rm_depend_args=()
dpo_depend_args=()
rloo_depend_spec=""

if [[ -n "${SFT_JOB_ID}" ]]; then
  rm_depend_args=(-W "depend=afterok:${SFT_JOB_ID}")
  dpo_depend_args=(-W "depend=afterok:${SFT_JOB_ID}")
  rloo_depend_spec="${SFT_JOB_ID}:"
fi

rm_job_id="$(qsub -v "${QSUB_ENV_VARS}" "${rm_depend_args[@]}" "${SCRIPT_DIR}/run_rm_qwen35_2b.pbs")"
dpo_job_id="$(qsub -v "${QSUB_ENV_VARS}" "${dpo_depend_args[@]}" "${SCRIPT_DIR}/run_dpo_qwen35_2b.pbs")"
rloo_job_id="$(qsub -v "${QSUB_ENV_VARS}" -W "depend=afterok:${rloo_depend_spec}${rm_job_id}" "${SCRIPT_DIR}/run_rloo_qwen35_2b.pbs")"

{
  echo "submission_time=$(date -Iseconds)"
  echo "profile=${MEDICAL_CLOSED_LOOP_PROFILE}"
  echo "sft_job_id=${SFT_JOB_ID}"
  echo "rm_job_id=${rm_job_id}"
  echo "dpo_job_id=${dpo_job_id}"
  echo "rloo_job_id=${rloo_job_id}"
  echo "data_root=${MEDICAL_CLOSED_LOOP_ROOT}"
  echo "eval_file=${MEDICAL_CLOSED_LOOP_EVAL_FILE}"
  echo "pbs_log_root=${PBS_LOG_ROOT}"
  echo "artifact_root=${ARTIFACT_ROOT}"
} | tee "${submission_log}"
