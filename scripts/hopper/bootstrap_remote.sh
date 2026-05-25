#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

ensure_common_dirs
module load "${APPTAINER_MODULE}"

if [[ ! -f "${IMAGE_SIF}" ]]; then
  apptainer pull "${IMAGE_SIF}" "${CONTAINER_IMAGE_URI}"
fi

BOOTSTRAP_STATE_FILE="${VENV_DIR}/.bootstrap_state"
BOOTSTRAP_STATE_VALUE="image=$(basename "${IMAGE_SIF}")|venv=system-site-packages|torch=container"

if [[ ! -f "${BOOTSTRAP_STATE_FILE}" ]] && [[ -d "${VENV_DIR}" ]]; then
  rm -rf "${VENV_DIR}"
fi

if [[ -f "${BOOTSTRAP_STATE_FILE}" ]] && [[ "$(cat "${BOOTSTRAP_STATE_FILE}")" != "${BOOTSTRAP_STATE_VALUE}" ]]; then
  rm -rf "${VENV_DIR}"
fi

apptainer exec --bind "$(apptainer_bind_paths)" "${IMAGE_SIF}" bash -lc "
set -euo pipefail
if [[ ! -x '${VENV_DIR}/bin/python' ]]; then
  rm -rf '${VENV_DIR}'
  python -m venv --system-site-packages '${VENV_DIR}'
fi
source '${VENV_DIR}/bin/activate'
python -m pip install --upgrade pip 'setuptools<82' wheel
python -m pip install -r '${PROJECT_ROOT}/requirements.txt'
python -m pip install 'gradio>=3.50.2'
python '${PROJECT_ROOT}/scripts/hopper/bootstrap_smoke.py'
printf '%s' '${BOOTSTRAP_STATE_VALUE}' > '${BOOTSTRAP_STATE_FILE}'
"

echo "[bootstrap_remote] image=${IMAGE_SIF}"
echo "[bootstrap_remote] venv=${VENV_DIR}"
