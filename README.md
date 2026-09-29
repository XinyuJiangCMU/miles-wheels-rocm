# miles-wheels-rocm

Prebuilt ROCm / gfx950 (MI350X / MI355X) wheels and binaries for the Miles training image.
Build recipes live in this repository; artifacts are attached to GitHub Releases
rather than committed to git.

## Releases

### `rocm724-gfx950-v0.5.20` (current)

GPU-dependent artifacts were rebuilt inside the exact target base
`rocm/sgl-dev:v0.5.20-rocm724-mi35x-20260919`
(`sha256:e85389543d3a850ca0f94c541b00581bc4bddf5d0d172deac228a76da167f18a`):
Ubuntu 24.04, Python 3.12, torch 2.11.0+rocm7.2, ROCm 7.2.4, gfx950.

- `transformer_engine-2.18.0.dev0-cp312-cp312-linux_x86_64.whl` is rebuilt for
  this base from `XinyuJiangCMU/TransformerEngine@2f663a0b87580ae375894bf42b9de87b1edc9b31`,
  the same fixed TE source used by the ROCm 10 shelf; SHA256
  `87e308199047c78d1eb6fce74144ae9c6670193f53dd46305e26d1359d2bd73d`.
- `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` is built from the PyPI 2.8.3
  sdist with `GPU_ARCHS=gfx950 BUILD_TARGET=rocm`.
- `sglang_router-0.3.2-cp38-abi3-manylinux_2_39_x86_64.whl` and
  `sgl-model-gateway-linux-x86_64.tar.gz` were rebuilt on 2026-09-28 from
  `radixark/sgl-router-for-miles@0e7c1ac7842317f39a9276af36f5f3a68ed85be8`,
  including the `lora_backfill_paths` forwarding fix.
- `libhsa-runtime64.so.1.18.70204.vmmfix` is built from
  `ROCm/ROCR-Runtime@e5498ba92dad7099d2027bd22bd7295ca1caf833` (`rocm-7.2.4`)
  plus `rocr-vmm-pause-fix-7.2.patch`. A stock/candidate A/B with the Miles-pinned
  `torch_memory_saver` showed stock freeing zero bytes and the candidate freeing
  and restoring 1,000,341,504 bytes while preserving the virtual address.

The base already contains Apex `1.10.0+rocm7.2.4.git751f5dd5`; Apex is validated
in place and deliberately not rebuilt or shipped on this shelf. The release notes
record each artifact's exact SHA256.

### `rocm10-gfx950-v0.5.18`

The ROCm 10 / Python 3.12 shelf used by the Miles `rocm10-mi35x` image carries
`transformer_engine-2.18.0.dev0-cp312-cp312-linux_x86_64.whl`. It was built in
`rocm/sgl-dev:miles-rocm10-mi35x-20260912` from
`XinyuJiangCMU/TransformerEngine@miles-dev`
(`2f663a0b87580ae375894bf42b9de87b1edc9b31`) with the Miles fp32-accum wgrad and
CP softmax-LSE dynamic-shape fixes. SHA256:
`5153c4119f2ac3e123d3c20f982e62a600e37ea56f00acd0aa9fb32e4fc55a54`.

Build this exact wheel inside the ROCm 10 target base with:

```bash
python build_te_wheel.py --out /out
```

The wheel does not set any `NVTE_*` runtime backend variable. Backend selection remains
opt-in per workload.

The router wheel and gateway were also refreshed on 2026-09-28 from the same router
commit listed above. This release uses the `manylinux_2_34_x86_64` router wheel.
`build_sglang_gateway.py` defaults to that published source commit for both releases.

### Historical releases

- `rocm720-gfx950-v0.5.16` targets
  `sgl-dev:v0.5.16-rocm720-mi35x-20260730` (Python 3.10, torch 2.9.1+rocm7.2.0).
  It is retained only for reproducibility; new Miles images must not use it.
- `rocm720-gfx950-v0.5.14` is the superseded v0.5.14 shelf.
- `rocm700-gfx950-v0.5.14` is the ROCm 7.0 counterpart.

The Miles Dockerfile selects the shelf with `WHEELS_TAG_ROCM` and verifies every
downloaded asset before installation.

## Build scripts

- `in-container-build.sh` builds Transformer Engine and flash-attn inside the exact
  target base. It emits wheels to `/out` and does not require a GPU.
- `build_te_wheel.py` builds the Miles Transformer Engine fork at a fixed full commit.
- `build_apex_wheel.py` builds ROCm Apex's native ops for an explicit `gfx942` or
  `gfx950` target. See the matching-GPU requirement below.
- `build_sglang_gateway.py` builds the sgl-router wheel and gateway binary at a fixed
  full commit using maturin and cargo.
- `build_rocr_vmmfix.py` rebuilds the matching point-release `libhsa-runtime64` with
  the VMM-pause patch. For 7.2.4 it emits `.1.18.70204.vmmfix`; never substitute the
  older `.70200` artifact.

Examples inside the matching target base:

```bash
python build_te_wheel.py --out /out
python build_sglang_gateway.py --out /out
python build_rocr_vmmfix.py --rocr-ref rocm-7.2.4 --out /out
GPU_ARCHS=gfx950 BUILD_TARGET=rocm pip wheel flash-attn==2.8.3 \
  --no-deps --no-build-isolation -w /out -v
```

Create a new immutable release for a new base or ABI. Do not point production
Dockerfiles at a mutable branch.

## Build Apex for ROCm 10

The existing ROCm 10 gfx950 Apex wheel uses
`ROCm/apex@40608ba22ccdb87e9f649600b0dc69d1f718d545`. The script defaults to that
same source and sets `APEX_BUILD_CPP_OPS=1 APEX_BUILD_CUDA_OPS=1`, which prebuild
the compatible native ops, including Miles' `fused_weight_gradient_mlp_cuda`.

Run inside the **exact SGLang base used by the target Miles image**, with its
existing ROCm PyTorch. For the MI300X / MI325X configuration, that base is
`rocm/sgl-dev:v0.5.20-rocm10-mi30x-20260919`. Record the pulled image digest with
the build results. Expose matching GPUs (`--device=/dev/kfd --device=/dev/dri`
and the required video/render group permissions), mount this repository at
`/workspace/miles-wheels-rocm`, and mount a persistent output directory at `/out`.

This Apex version probes `rocminfo` and can override `PYTORCH_ROCM_ARCH` with the
detected hardware. Therefore **build gfx942 on MI300X / MI325X, and gfx950 on
MI350X / MI355X**. The script rejects missing or mismatched devices. It does not
claim GPU-free cross-compilation support.

Inside that container:

```bash
apt-get update
apt-get install -y build-essential git ninja-build
python -m pip install setuptools wheel packaging cxxfilt==0.3.0 py-cpuinfo==9.0.0
cd /workspace/miles-wheels-rocm
set -o pipefail
python build_apex_wheel.py --gpu-arch gfx942 --out /out/gfx942 --jobs 32 \
  2>&1 | tee /out/apex-gfx942-build.log
```

For MI350X / MI355X, use the matching MI35X base, `--gpu-arch gfx950`, and a
separate output directory. `MAX_JOBS` supplies the default job limit; `--jobs`
overrides it. Neither pip dependencies nor a different PyTorch are installed by
the build script itself.

The output contains the wheel and a `.build.json` receipt recording the full
source commit, Python/PyTorch/HIP versions, target GPU, build flags and SHA256.
The script checks that the wheel contains the native wgrad extension; a wheel
containing only Python JIT wrappers is rejected. It does not upload artifacts or
install the wheel. Successful compilation is not GPU runtime validation: install
the wheel in a fresh matching container and run the native op before publishing.
The new gfx942 build path still needs that validation.

This is the Apex recipe only. `build_te_wheel.py` and `in-container-build.sh`
currently target gfx950; they are not yet a complete MI300 wheel build entry point.
