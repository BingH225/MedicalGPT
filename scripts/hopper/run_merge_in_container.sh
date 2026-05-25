#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <base_model> <tokenizer_path> <lora_model> <output_dir>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

BASE_MODEL_PATH="$1"
TOKENIZER_PATH="$2"
LORA_MODEL_PATH="$3"
OUTPUT_DIR="$4"

module load "${APPTAINER_MODULE}"
normalize_cuda_visible_devices

apptainer exec --nv --bind "$(apptainer_bind_paths)" "${IMAGE_SIF}" bash -lc "
set -euo pipefail
$(container_env_exports)
source '${VENV_DIR}/bin/activate'
cd '${PROJECT_ROOT}'
rm -rf '${OUTPUT_DIR}'
python -u tools/merge_peft_adapter.py \
  --base_model '${BASE_MODEL_PATH}' \
  --tokenizer_path '${TOKENIZER_PATH}' \
  --lora_model '${LORA_MODEL_PATH}' \
  --output_dir '${OUTPUT_DIR}'
"
