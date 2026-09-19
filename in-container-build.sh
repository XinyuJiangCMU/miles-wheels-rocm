#!/usr/bin/env bash
# Runs INSIDE the exact target rocm/sgl-dev base. Builds the Miles TE 2.17 fork
# and flash-attn 2.8.3, emitting wheels into /out instead of installing them.
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
apt-get install -y build-essential cmake git rocm-llvm-dev xxd

echo "=== [2/3] Transformer Engine 2.18.0.dev0 (Miles fork) -> wheel ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/build_te_wheel.py" --out /out

echo "=== [3/3] flash-attn 2.8.3 (rocm) -> wheel ==="
GPU_ARCHS=gfx950 BUILD_TARGET=rocm pip wheel flash-attn==2.8.3 --no-deps --no-build-isolation -w /out -v

echo "=== DONE. wheels in /out: ==="
ls -la /out/*.whl
