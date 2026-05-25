#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <sft_job_id>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

SFT_JOB_ID="$1"
RUN_DIR="${OUTPUT_ROOT}/sft_${SFT_JOB_ID}"
ADAPTER_DIR="${RUN_DIR}/adapter"
MERGED_DIR="${ARTIFACT_ROOT}/merged/sft-qwen35-2b"

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Adapter dir not found: ${ADAPTER_DIR}" >&2
  exit 1
fi

ensure_common_dirs
module load "${APPTAINER_MODULE}"
normalize_cuda_visible_devices

apptainer exec --nv --bind "$(apptainer_bind_paths)" "${IMAGE_SIF}" bash -lc "
set -euo pipefail
$(container_env_exports)
source '${VENV_DIR}/bin/activate'
cd '${PROJECT_ROOT}'
rm -rf '${MERGED_DIR}'
python -u tools/merge_peft_adapter.py \
  --base_model '${BASE_MODEL}' \
  --tokenizer_path '${BASE_MODEL}' \
  --lora_model '${ADAPTER_DIR}' \
  --output_dir '${MERGED_DIR}'
" | tee "${RUN_DIR}/merge_retry.log"
