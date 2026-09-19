# miles-wheels-rocm

Prebuilt ROCm / gfx950 (MI355X) wheels and binaries for the Miles training image.
Build recipes live in this repository; artifacts are attached to GitHub Releases
rather than committed to git.

## Releases

### `rocm724-gfx950-v0.5.20` (current)

All artifacts were rebuilt inside the exact target base
`rocm/sgl-dev:v0.5.20-rocm724-mi35x-20260919`
(`sha256:e85389543d3a850ca0f94c541b00581bc4bddf5d0d172deac228a76da167f18a`):
Ubuntu 24.04, Python 3.12, torch 2.11.0+rocm7.2, ROCm 7.2.4, gfx950.

- `transformer_engine-2.17.0-cp312-cp312-linux_x86_64.whl` is built from
  `JessicaJiang-123/TransformerEngine@58109c88cb277d7f7763d239b7cbadfbe77ff241`.
- `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` is built from the PyPI 2.8.3
  sdist with `GPU_ARCHS=gfx950 BUILD_TARGET=rocm`.
- `sglang_router-0.3.2-cp38-abi3-manylinux_2_39_x86_64.whl` and
  `sgl-model-gateway-linux-x86_64.tar.gz` are built from
  `radixark/sgl-router-for-miles@a2ad8d0c84191efea67e1bb2b61d0c634b84c2ce`.
- `libhsa-runtime64.so.1.18.70204.vmmfix` is built from
  `ROCm/ROCR-Runtime@e5498ba92dad7099d2027bd22bd7295ca1caf833` (`rocm-7.2.4`)
  plus `rocr-vmm-pause-fix-7.2.patch`. A stock/candidate A/B with the Miles-pinned
  `torch_memory_saver` showed stock freeing zero bytes and the candidate freeing
  and restoring 1,000,341,504 bytes while preserving the virtual address.

The base already contains Apex `1.10.0+rocm7.2.4.git751f5dd5`; Apex is validated
in place and deliberately not rebuilt or shipped on this shelf. See the release's
`SHA256SUMS.rocm724-gfx950-v0.5.20` for exact artifact hashes.

### Historical releases

- `rocm720-gfx950-v0.5.16` targets
  `sgl-dev:v0.5.16-rocm720-mi35x-20260730` (Python 3.10, torch 2.9.1+rocm7.2.0).
  It is retained only for reproducibility; new Miles images must not use it.
- `rocm720-gfx950-v0.5.14` is the superseded v0.5.14 shelf.
- `rocm700-gfx950-v0.5.14` is the ROCm 7.0 counterpart.

The Miles Dockerfile selects the current shelf with
`--build-arg WHEELS_TAG_ROCM=rocm724-gfx950-v0.5.20` and verifies every downloaded
asset before installation.

## Build scripts

- `in-container-build.sh` builds Transformer Engine and flash-attn inside the exact
  target base. It emits wheels to `/out` and does not require a GPU.
- `build_te_wheel.py` builds the Miles Transformer Engine fork at a fixed full commit.
- `build_sglang_gateway.py` builds the sgl-router wheel and gateway binary at a fixed
  full commit using maturin and cargo.
- `build_rocr_vmmfix.py` rebuilds the matching point-release `libhsa-runtime64` with
  the VMM-pause patch. For 7.2.4 it emits `.1.18.70204.vmmfix`; never substitute the
  older `.70200` artifact.

Example inside the target base:

```bash
python build_te_wheel.py --out /out
python build_sglang_gateway.py --out /out
python build_rocr_vmmfix.py --rocr-ref rocm-7.2.4 --out /out
GPU_ARCHS=gfx950 BUILD_TARGET=rocm pip wheel flash-attn==2.8.3 \
  --no-deps --no-build-isolation -w /out -v
```

Create a new immutable release for a new base or ABI. Do not overwrite an existing
release asset and do not point production Dockerfiles at a mutable branch.
