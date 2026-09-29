#!/usr/bin/env python3
"""Build the ROCm Apex wheel, including the native ops required by Miles.

Run inside the target SGLang runtime base with matching GPUs exposed. The pinned
Apex builder probes rocminfo and can override PYTORCH_ROCM_ARCH, so this script
requires matching hardware rather than promising GPU-free cross-compilation.

Examples (inside the matching base container):
    python build_apex_wheel.py --gpu-arch gfx942 --out /out/gfx942
    python build_apex_wheel.py --gpu-arch gfx950 --out /out/gfx950 --jobs 16

Produces a wheel and a .build.json receipt with its SHA256 and build environment.
Does not install the wheel or upload it. GPU runtime validation is a separate step.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile

APEX_REPO = "https://github.com/ROCm/apex.git"
APEX_COMMIT = "40608ba22ccdb87e9f649600b0dc69d1f718d545"


def run(cmd, *, cwd=None, env=None):
    print(f"+ {shlex.join([str(arg) for arg in cmd])}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def validate_gpu(arch, rocminfo):
    detected = set(re.findall(r"\bgfx[0-9a-f]+\b", rocminfo))
    if detected != {arch}:
        raise RuntimeError(
            f"Requested {arch}, but rocminfo reports {sorted(detected)}. "
            "Use matching hardware and expose only that GPU architecture."
        )
    waves = set(re.findall(r"Wavefront Size:\s+(\d+)", rocminfo))
    if waves != {"64"}:
        raise RuntimeError(f"Expected wavefront size 64 for {arch}, got {sorted(waves)}")


def check_environment(arch):
    for tool in ("git", "c++", "ninja"):
        if not shutil.which(tool):
            raise RuntimeError(f"Missing {tool}; see README's Apex build prerequisites.")
    import torch
    from torch.utils.cpp_extension import ROCM_HOME

    if not torch.version.hip or not ROCM_HOME:
        raise RuntimeError("Use the target ROCm/PyTorch runtime base, not a CPU or CUDA environment.")
    if not (Path(ROCM_HOME) / "bin/hipcc").is_file():
        raise RuntimeError(f"HIP compiler missing under {ROCM_HOME}")
    # Match the executable chosen by Apex's op_builder/builder.py.
    rocminfo = Path("/opt/rocm/bin/rocminfo")
    if not rocminfo.is_file():
        rocminfo = shutil.which("rocminfo")
    if not rocminfo:
        raise RuntimeError("rocminfo is required by this Apex source version.")
    output = subprocess.check_output([str(rocminfo)], text=True)
    validate_gpu(arch, output)
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch cannot access a GPU; check container device permissions.")
    return {"python": platform.python_version(), "torch": torch.__version__,
            "hip": torch.version.hip, "rocm_home": str(ROCM_HOME), "gpu_arch": arch}


def validate_wheel(wheel):
    with zipfile.ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"Corrupt wheel: {wheel}")
        native = [name for name in archive.namelist()
                  if name.startswith("apex/fused_weight_gradient_mlp_cuda") and name.endswith(".so")]
        if not native:
            raise RuntimeError("Apex wheel is missing the native fused_weight_gradient_mlp_cuda op.")
    return native


def build(args):
    environment = check_environment(args.gpu_arch)
    flags = {name: args.gpu_arch for name in
             ("GPU_ARCH", "GPU_ARCHS", "PYTORCH_ROCM_ARCH", "GPU_ARCH_LIST", "AMDGPU_TARGET")}
    flags.update(APEX_BUILD_CPP_OPS="1", APEX_BUILD_CUDA_OPS="1", MAX_JOBS=str(args.jobs))
    env = {**os.environ, **flags, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A unique checkout avoids deleting or reusing anyone's /tmp/apex tree.
    with tempfile.TemporaryDirectory(prefix=".apex-build-", dir=out) as tmp:
        source = Path(tmp) / "source"
        source.mkdir()
        run(["git", "init", "-q"], cwd=source)
        run(["git", "remote", "add", "origin", APEX_REPO], cwd=source)
        run(["git", "fetch", "--depth=1", "origin", args.commit], cwd=source)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if commit != args.commit:
            raise RuntimeError(f"Unexpected source commit: {commit}")
        run(["git", "submodule", "update", "--init", "--recursive", "--depth=1"], cwd=source)
        wheel_dir = Path(tmp) / "wheels"
        run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation",
             "--no-cache-dir", "--wheel-dir", str(wheel_dir), "-v"], cwd=source, env=env)
        wheels = list(wheel_dir.glob("apex-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one newly built Apex wheel, found {len(wheels)}")
        wheel = wheels[0]
        native = validate_wheel(wheel)
        hasher = hashlib.sha256()
        with wheel.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        target = out / wheel.name
        if target.exists():
            raise FileExistsError(f"Output already exists: {target}; use a new output directory.")
        shutil.copy2(wheel, target)
        receipt = {"source_repo": APEX_REPO, "source_commit": commit, **environment,
                   "build_flags": flags, "wheel": target.name, "sha256": digest,
                   "native_ops": native, "gpu_runtime_validation": "not run"}
        target.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Built {target}\nSHA256: {digest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-arch", choices=("gfx942", "gfx950"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="Separate output directory for this base and GPU arch")
    parser.add_argument("--commit", default=APEX_COMMIT, help="Full Apex source commit SHA")
    parser.add_argument("--jobs", type=int, default=os.environ.get("MAX_JOBS", "32"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit must be a full, lowercase 40-character commit SHA")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build(args)


if __name__ == "__main__":
    main()
