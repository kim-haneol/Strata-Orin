#!/usr/bin/env bash
# Build the preserved Orin engine and its validation targets. Does not run inference.
set -euo pipefail
src=${1:?usage: build.sh STRATA_SOURCE BUILD_DIR PATCHED_GGML_SOURCE}
build=${2:?build directory}
ggml=${3:?patched GGML source directory}
nvcc=${CUDACXX:-/usr/local/cuda/bin/nvcc}
cmake -S "$src" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=OFF \
    -DSTRATA_GGML_DIR="$ggml" -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_CUDA_COMPILER="$nvcc"
targets=(strata strata-device native_expert_parity native_hadamard_test
         router_dot_parity gdn_parity gdn_rec_parity gemm_bf16_parity gr_parity
         dequant_bf16_test)
# This validation target was added with the final optimization, after the ARM64 base.
if [[ -f "$src/tests/cuda/mmq_gather_group_test.cpp" ]]; then
    targets+=(mmq_gather_group_test)
fi
# The extension adds checkpoint policy and pair-trellis validation targets.
if [[ -f "$src/src/program/conv_cache_test.cpp" ]]; then
    targets+=(conv_cache_test)
fi
if [[ -f "$src/tests/cuda/prefill_tqp_parity.cpp" ]]; then
    targets+=(prefill_tqp_parity)
fi
cmake --build "$build" -j "${GYRO_BUILD_JOBS:-4}" --target "${targets[@]}"
