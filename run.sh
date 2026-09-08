#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
bash "${ROOT}/build.sh"
python3 "${ROOT}/run.py"
