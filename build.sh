#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# Load CANN (set_env may touch unbound vars; disable -u temporarily)
set +u
if [ -f "${ASCEND_HOME_PATH:-}/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "${ASCEND_HOME_PATH}/set_env.sh"
elif [ -f /home/developer/Ascend/cann-9.0.0/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /home/developer/Ascend/cann-9.0.0/set_env.sh
elif [ -f /home/developer/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /home/developer/Ascend/ascend-toolkit/set_env.sh
elif [ -f /home/developer/Ascend/cann/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /home/developer/Ascend/cann/set_env.sh
else
  echo "ERROR: CANN set_env.sh not found. Set ASCEND_HOME_PATH first."
  exit 1
fi
set -u

MODE="release"
for arg in "$@"; do
  case "$arg" in
    --debug|--onboard|--cpudebug|--simulator|--mssanitizer|-u) MODE="debug" ;;
  esac
done

export SOC_VERSION="${SOC_VERSION:-ascend910}"
export NPU_ARCH="${NPU_ARCH:-dav-2201}"

echo "=== Build rotaryposemb (mode=${MODE}, NPU_ARCH=${NPU_ARCH}) ==="
rm -rf build
mkdir -p build
cd build

CMAKE_ARGS=(-DNPU_ARCH="${NPU_ARCH}")
if [ "${MODE}" = "debug" ]; then
  CMAKE_ARGS+=(-DCMAKE_BUILD_TYPE=Debug)
else
  CMAKE_ARGS+=(-DCMAKE_BUILD_TYPE=Release)
fi

cmake .. "${CMAKE_ARGS[@]}"
make -j"$(nproc 2>/dev/null || echo 4)"

echo "=== Build OK: ${ROOT}/build/rotary_pos_emb_custom ==="
