#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."
ROBOTWIN_REPO_ROOT="${ROBOTWIN_REPO_ROOT:-}"
ROBOTWIN_ENV_SOURCE="${ROBOTWIN_ENV_SOURCE:-${REPO_ROOT}/.venv-robotwin/bin/activate}"
imagewam_require_env ROBOTWIN_REPO_ROOT
if [ ! -d "${ROBOTWIN_REPO_ROOT}/script" ]; then
  echo "Invalid ROBOTWIN_REPO_ROOT (missing script/): ${ROBOTWIN_REPO_ROOT}" >&2
  exit 2
fi
if [ -f "${ROBOTWIN_ENV_SOURCE}" ]; then
  # shellcheck disable=SC1090
  source "${ROBOTWIN_ENV_SOURCE}"
fi
(cd "${ROBOTWIN_REPO_ROOT}" && imagewam_run bash script/_install_uv.sh)
(cd "${ROBOTWIN_REPO_ROOT}" && imagewam_run bash script/_download_assets.sh)
