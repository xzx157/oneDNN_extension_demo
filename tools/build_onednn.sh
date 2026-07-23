#!/usr/bin/env bash
set -euo pipefail

DNNL_VERSION="${DNNL_VERSION:-v3.8.1}"
DNNL_PREFIX="${DNNL_ROOT:-/opt/onednn}"
WORK_DIR="${DNNL_BUILD_DIR:-/tmp/onednn-build}"
ARCHIVE="${WORK_DIR}/onednn.tar.gz"
SOURCE_DIR="${WORK_DIR}/src"
BUILD_DIR="${WORK_DIR}/build"

mkdir -p "${WORK_DIR}" "${SOURCE_DIR}" "${BUILD_DIR}"
curl --fail --location --retry 3 \
  "https://github.com/uxlfoundation/oneDNN/archive/refs/tags/${DNNL_VERSION}.tar.gz" \
  --output "${ARCHIVE}"
tar --extract --gzip --file "${ARCHIVE}" --strip-components=1 \
  --directory "${SOURCE_DIR}"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${DNNL_PREFIX}" \
  -DDNNL_BUILD_TESTS=OFF \
  -DDNNL_BUILD_EXAMPLES=OFF \
  -DDNNL_ENABLE_WORKLOAD=INFERENCE
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
cmake --install "${BUILD_DIR}"

test -f "${DNNL_PREFIX}/include/oneapi/dnnl/dnnl.hpp"
find "${DNNL_PREFIX}" -type f -name 'libdnnl.so*' -print -quit | grep -q .
