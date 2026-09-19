#!/usr/bin/env python3
"""Rebuild libhsa-runtime64 (ROCr) with the ROCm 7.2 VMM-pause fix.

On ROCm 7.2, ROCr's ``~MappedHandleAllowedAgent()`` CPU-agent branch early-returns
without unmapping the imported dma-buf, so ``hipMemUnmap`` + ``hipMemRelease`` leak the
physical HBM while the VA stays reserved (ROCm/ROCm#6021). ``torch_memory_saver.pause()``
then frees nothing and colocated RL OOMs. The one-function upstream fix
(ROCm/rocm-systems#4363, merge ``e27ce55c``) is only on ``develop`` — not in any released
7.2.x (checked through 7.2.4) — so this rebuilds ``libhsa-runtime64`` from the
matching ROCm tag with that fix rebased on top (``rocr-vmm-pause-fix-7.2.patch``).

The output name includes the point-release build number, for example
``libhsa-runtime64.so.1.18.70200.vmmfix`` for ROCm 7.2.0 and
``libhsa-runtime64.so.1.18.70204.vmmfix`` for ROCm 7.2.4. It is
framework-agnostic, but must only replace the exact matching runtime.

Prereqs: matching ROCm install, ``rocm-llvm-dev`` (trap-handler Clang cmake
config), ``xxd``, cmake, git, and patch.

Usage (standalone, inside the exact target ROCm base container):
    python build_rocr_vmmfix.py --out /tmp/wheels
    python build_rocr_vmmfix.py --rocr-ref rocm-7.2.4 --out /tmp/wheels
"""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

ROCR_REPO_DEFAULT = "https://github.com/ROCm/ROCR-Runtime.git"
ROCR_REF_DEFAULT = "rocm-7.2.4"
PATCH_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rocr-vmm-pause-fix-7.2.patch")


@dataclass
class BuildConfig:
    repo: str = ROCR_REPO_DEFAULT
    ref: str = ROCR_REF_DEFAULT
    patch: str = PATCH_DEFAULT
    rocm_path: str = "/opt/rocm"
    rocm_patch_version: str | None = None


def _patch_version_from_ref(ref: str) -> str:
    """Convert a release ref such as rocm-7.2.4 to ROCM_PATCH_VERSION=70204."""
    match = re.fullmatch(r"rocm-(\d+)\.(\d+)\.(\d+)", ref)
    if not match:
        raise RuntimeError(
            f"cannot derive ROCM_PATCH_VERSION from {ref!r}; "
            "pass --rocm-patch-version explicitly"
        )
    major, minor, patch = map(int, match.groups())
    return str(major * 10000 + minor * 100 + patch)


def _run(cmd, *, env=None, cwd=None):
    merged_env = {**os.environ, **(env or {})}
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}\n", flush=True)
    result = subprocess.run(cmd, env=merged_env, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _ensure_prereqs():
    for tool in ("git", "cmake"):
        if not _command_exists(tool):
            raise RuntimeError(f"{tool} is required.")
    # rocm-llvm-dev provides the Clang cmake config the trap handler needs.
    if not os.path.isdir("/opt/rocm/llvm/lib/cmake"):
        print("[warn] /opt/rocm/llvm/lib/cmake missing — install rocm-llvm-dev if configure fails.")


def build(cfg: BuildConfig, out_dir: str):
    """Clone ROCR-Runtime @ ref, apply the fix, build hsa-runtime64, copy the .so to out_dir."""
    src = "/tmp/ROCR-Runtime"
    _ensure_prereqs()
    if not os.path.isfile(cfg.patch):
        raise RuntimeError(f"patch not found: {cfg.patch}")
    if os.path.exists(src):
        shutil.rmtree(src)
    os.makedirs(out_dir, exist_ok=True)
    patch_version = cfg.rocm_patch_version or _patch_version_from_ref(cfg.ref)
    soname = f"libhsa-runtime64.so.1.18.{patch_version}"
    out_name = f"{soname}.vmmfix"

    try:
        _run(["git", "clone", "--depth", "1", "-b", cfg.ref, cfg.repo, src])
        _run(["git", "log", "-1", "--format=source_commit=%H"], cwd=src)
        _run(["git", "apply", "--stat", cfg.patch], cwd=src)  # show what it touches
        _run(["patch", "-p1", "-i", cfg.patch], cwd=src)
        _run([
            "cmake", "-S", ".", "-B", "build",
            f"-DCMAKE_INSTALL_PREFIX={cfg.rocm_path}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DROCM_PATCH_VERSION={patch_version}",
            f"-DCMAKE_PREFIX_PATH={cfg.rocm_path};{cfg.rocm_path}/llvm",
        ], cwd=src)
        jobs = os.environ.get("MAX_JOBS", str(os.cpu_count() or 4))
        _run(["cmake", "--build", "build", "--target", "hsa-runtime64", "-j", jobs], cwd=src)

        built = os.path.join(src, "build", "rocr", "lib", soname)
        if not os.path.isfile(built):
            raise RuntimeError(f"expected build output missing: {built}")

        dynamic = subprocess.run(["readelf", "-d", built], capture_output=True, text=True, check=True)
        if "[libhsa-runtime64.so.1]" not in dynamic.stdout:
            raise RuntimeError("built ROCr has an unexpected DT_SONAME")

        # sanity: the fix pulls in rocr::os::UncommitMemory — confirm the symbol is referenced.
        nm = subprocess.run(["nm", "-D", built], capture_output=True, text=True)
        if "UncommitMemory" not in nm.stdout:
            print("[warn] UncommitMemory not found in dynamic symbols (stripped build is OK; verify via objdump).")

        dst = os.path.join(out_dir, out_name)
        shutil.copy2(built, dst)
        print(f"\nBuilt vmmfix ROCr: {dst}")
        print("Install it over $(readlink -f /opt/rocm/lib/libhsa-runtime64.so.1) — overwrite in place,")
        print("stash the stock .so OUTSIDE the lib dir, and do NOT run ldconfig (a same-soname backup")
        print("in the lib dir would steal the .so.1 symlink).")
    finally:
        shutil.rmtree(src, ignore_errors=True)


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="/tmp/wheels", help="Output directory for the .so")
    p.add_argument("--repo", default=ROCR_REPO_DEFAULT, help="ROCR-Runtime git repository")
    p.add_argument("--rocr-ref", default=ROCR_REF_DEFAULT, help="ROCR-Runtime tag/branch (match your ROCm point release)")
    p.add_argument(
        "--rocm-patch-version",
        default=None,
        help="numeric ROCM_PATCH_VERSION; derived from refs such as rocm-7.2.4 by default",
    )
    p.add_argument("--patch", default=PATCH_DEFAULT, help="path to rocr-vmm-pause-fix patch")
    args = p.parse_args()

    build(
        BuildConfig(
            repo=args.repo,
            ref=args.rocr_ref,
            patch=args.patch,
            rocm_patch_version=args.rocm_patch_version,
        ),
        args.out,
    )


if __name__ == "__main__":
    main()
