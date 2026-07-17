# miles-wheels-rocm

Prebuilt ROCm / gfx950 (MI355X) wheels + binaries for the miles training image
(`docker/Dockerfile.rocm720`), mirroring the NV `yueming-yuan/miles-wheels` flow: the
build scripts live here, and the artifacts are attached to **Releases** (not committed).

## Releases (artifacts are release assets)

- **rocm720-gfx950-v0.5.14** — built on the `sgl-dev v0.5.14-rocm720-mi35x-20260627` base
  (torch 2.9.1+rocm7.2.0, Python 3.10):
  - `flash_attn-2.8.3-...whl` — flash-attn 2.8.3 (PyPI) via `pip wheel --no-build-isolation`,
    `GPU_ARCHS=gfx950 BUILD_TARGET=rocm`. Links libamdhip64 / libMIOpen.
  - `sglang_router-0.3.2-...whl` + `sgl-model-gateway-linux-x86_64.tar.gz` — from
    `radixark/sgl-router-for-miles @ a2ad8d0` via maturin + cargo (vendored-openssl).
  - `libhsa-runtime64.so.1.18.70200.vmmfix` — ROCr rebuilt from the `rocm-7.2.0` tag with the
    ROCm 7.2 VMM-pause fix (ROCm/rocm-systems#4363, merge `e27ce55c`), which is only on `develop`
    and not in any released 7.2.x. Fixes `torch_memory_saver.pause()` freeing 0 bytes on ROCm 7.2.
    Framework-agnostic (drop it into any ROCm 7.2.0 stack), but pinned to that soname. Built by
    `build_rocr_vmmfix.py`.
- **rocm700-gfx950-v0.5.14** — the ROCm 7.0 counterpart (flash-attn + transformer_engine).
  ROCm 7.0 has no VMM-pause regression, so no vmmfix asset here.

The Dockerfile selects a release with `--build-arg WHEELS_TAG_ROCM=rocm720-gfx950-v0.5.14`
and downloads all assets, then installs each. See its `SGL_ROUTER_USE_WHEELS` switch.

## Build scripts

- `build_sglang_gateway.py` — build the sgl-router wheel + gateway binary from source
  (rustup + maturin + cargo). `python build_sglang_gateway.py --out /tmp/wheels`.
- `in-container-build.sh` — helper to build the wheels inside a rocm720 base container.
- `build_rocr_vmmfix.py` — rebuild `libhsa-runtime64` with the ROCm 7.2 VMM-pause fix (clone
  ROCR-Runtime @ `rocm-7.2.0` + apply `rocr-vmm-pause-fix-7.2.patch` + cmake). Needs `rocm-llvm-dev`.
  `python build_rocr_vmmfix.py --out /tmp/wheels`. Delete once a released ROCm ships the fix.
- `build_te_wheel.py` — build the ROCm/gfx950 Transformer Engine wheel from the fp8 fork
  (`XinyuJiangCMU/TransformerEngine @ miles-dev`) with `NVTE_NO_LOCAL_VERSION=1` for a clean
  PEP440 version. Run inside a rocm720 base container: `python build_te_wheel.py --out /out`.

Upload the produced artifacts to a new Release tag, then point `WHEELS_TAG_ROCM` at it.
