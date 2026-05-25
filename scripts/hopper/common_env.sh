#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROJECT_NAME="${PROJECT_NAME:-medicalgpt}"
PROJECT_ROOT_REMOTE="${PROJECT_ROOT_REMOTE:-/scratch/e1561245/cot_yz/medicalgpt}"

RUNTIME_ROOT="${RUNTIME_ROOT:-${PROJECT_ROOT}/runtime/hopper}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/hopper}"
PBS_LOG_ROOT="${PBS_LOG_ROOT:-${OUTPUT_ROOT}/pbs_logs}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/hopper}"

HOPPER_SFT_DATA_DIR="${HOPPER_SFT_DATA_DIR:-${PROJECT_ROOT}/data/hopper_sft}"
HOPPER_REWARD_DATA_DIR="${HOPPER_REWARD_DATA_DIR:-${PROJECT_ROOT}/data/hopper_reward}"

HF_HOME="${HF_HOME:-/scratch/e1561245/hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"

APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer/1.3.1}"
CONTAINER_IMAGE_URI="${CONTAINER_IMAGE_URI:-docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
IMAGE_SIF="${IMAGE_SIF:-${RUNTIME_ROOT}/pytorch_2.5.1-cuda12.4-cudnn9-runtime.sif}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
SFT_MAX_TRAIN_SAMPLES="${SFT_MAX_TRAIN_SAMPLES:-1000}"
RM_MAX_TRAIN_SAMPLES="${RM_MAX_TRAIN_SAMPLES:-500}"
DPO_MAX_TRAIN_SAMPLES="${DPO_MAX_TRAIN_SAMPLES:-500}"
RLOO_MAX_STEPS="${RLOO_MAX_STEPS:-80}"

export PROJECT_ROOT
export PROJECT_NAME
export RUNTIME_ROOT
export OUTPUT_ROOT
export PBS_LOG_ROOT
export ARTIFACT_ROOT
export HOPPER_SFT_DATA_DIR
export HOPPER_REWARD_DATA_DIR
export HF_HOME
export HUGGINGFACE_HUB_CACHE
export TRANSFORMERS_CACHE
export APPTAINER_MODULE
export CONTAINER_IMAGE_URI
export IMAGE_SIF
export VENV_DIR
export BASE_MODEL
export SFT_MAX_TRAIN_SAMPLES
export RM_MAX_TRAIN_SAMPLES
export DPO_MAX_TRAIN_SAMPLES
export RLOO_MAX_STEPS

ensure_common_dirs() {
  mkdir -p "${RUNTIME_ROOT}" "${OUTPUT_ROOT}" "${PBS_LOG_ROOT}" "${ARTIFACT_ROOT}"
  mkdir -p "${HOPPER_SFT_DATA_DIR}" "${HOPPER_REWARD_DATA_DIR}"
}

normalize_cuda_visible_devices() {
  if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    IFS=',' read -ra dev_arr <<< "${CUDA_VISIBLE_DEVICES}"
    last_idx=$(( ${#dev_arr[@]} - 1 ))
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 "${last_idx}")"
  fi
}

apptainer_bind_paths() {
  printf "%s" "${PROJECT_ROOT}:${PROJECT_ROOT},/scratch/e1561245:/scratch/e1561245"
}

container_env_exports() {
  cat <<EOF
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
EOF
}
