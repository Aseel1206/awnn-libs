#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"

mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
cmake --install "${BUILD_DIR}"

echo
echo "Library build finished."
echo "Headers: ${INSTALL_DIR}/include"
echo "Library: ${INSTALL_DIR}/lib/libawnn_vision.so"
