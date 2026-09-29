# miles-wheels-rocm

Prebuilt ROCm wheels and binaries for Miles. Download artifacts from
[Releases](https://github.com/XinyuJiangCMU/miles-wheels-rocm/releases);
source revisions and checksums are recorded in the release notes.

- [ROCm 7.2.4 / gfx950](https://github.com/XinyuJiangCMU/miles-wheels-rocm/releases/tag/rocm724-gfx950-v0.5.20)
- [ROCm 10 / gfx950](https://github.com/XinyuJiangCMU/miles-wheels-rocm/releases/tag/rocm10-gfx950-v0.5.18)

## Build

Run inside the target Miles image's SGLang base container to match Python, PyTorch
and ROCm. Outputs go to `/out`; scripts do not upload them.

### Transformer Engine + flash-attn (gfx950)

```bash
bash in-container-build.sh
```

For TE only: `python build_te_wheel.py --out /out`. These scripts currently target gfx950.

### Router + gateway

```bash
apt-get update && apt-get install -y protobuf-compiler
python build_sglang_gateway.py --out /out
```

Use `--ref <commit>` to select a different router revision.

### Apex (ROCm 10)

Expose matching GPUs to the container: gfx942 for MI300X/MI325X, gfx950 for
MI350X/MI355X. The gfx942 build still needs validation on MI300 hardware.

```bash
apt-get update && apt-get install -y build-essential git ninja-build
python -m pip install setuptools wheel packaging cxxfilt==0.3.0 py-cpuinfo==9.0.0
python build_apex_wheel.py --gpu-arch gfx942 --out /out/gfx942 --jobs 32
```

For MI35X, use `--gpu-arch gfx950 --out /out/gfx950` in the matching base.
ROCm 7.2.4 uses Apex from its base image.

### ROCr VMM fix (ROCm 7.2.4)

```bash
apt-get update && apt-get install -y build-essential cmake git patch rocm-llvm-dev xxd
python build_rocr_vmmfix.py --rocr-ref rocm-7.2.4 --out /out
```
