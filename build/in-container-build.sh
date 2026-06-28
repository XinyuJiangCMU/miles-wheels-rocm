#!/usr/bin/env bash
# Runs INSIDE rocm/sgl-dev base. Builds ROCm wheels for TE v2.8_rocm + flash-attn 2.8.3
# the same way docker/Dockerfile.rocm builds them, but emits .whl into /out instead of installing.
# CPU cross-compile for gfx950 — no GPU needed.
set -euxo pipefail

export GPU_ARCH=gfx950 PYTORCH_ROCM_ARCH=gfx950 GPU_ARCH_LIST=gfx950 AMDGPU_TARGET=gfx950
export NVTE_FRAMEWORK=pytorch NVTE_ROCM_ARCH=gfx950 NVTE_USE_HIPBLASLT=1 NVTE_USE_ROCM=1
export CMAKE_PREFIX_PATH=/opt/rocm:/opt/rocm/hip:/usr/local:/usr
export MAX_JOBS=32
export PIP_ROOT_USER_ACTION=ignore

mkdir -p /out

echo "=== [1/3] apt build deps (same as Dockerfile.rocm) ==="
apt-get update
apt-get install -y build-essential cmake git

echo "=== [2/3] Transformer Engine v2.8_rocm -> wheel ==="
rm -rf /root/TransformerEngine
git clone --recursive --branch v2.8_rocm https://github.com/ROCm/TransformerEngine.git /root/TransformerEngine
cd /root/TransformerEngine
NVTE_FUSED_ATTN=0 pip wheel . --no-deps --no-build-isolation -w /out -v

echo "=== [3/3] flash-attn 2.8.3 (rocm) -> wheel ==="
cd /root
GPU_ARCHS=gfx950 BUILD_TARGET=rocm pip wheel flash-attn==2.8.3 --no-deps --no-build-isolation -w /out -v

echo "=== DONE. wheels in /out: ==="
ls -la /out/*.whl
