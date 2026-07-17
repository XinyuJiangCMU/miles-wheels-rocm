#!/usr/bin/env python3
"""Build the ROCm / gfx950 Transformer Engine wheel from the miles fp8 fork.

The published ``transformer_engine-<ver>.dev0.<sha>-cp310-...whl`` is built from
``XinyuJiangCMU/TransformerEngine`` (the fp8 fork, ``miles-dev`` integration branch),
NOT from upstream ``ROCm/TransformerEngine``. It is built with ``NVTE_NO_LOCAL_VERSION=1``
so the version is a clean PEP440 string (no ``+<sha>`` local tag, which GitHub Releases
mangle into an invalid filename); the source commit is tracked in the release / manifest.

This is a CPU cross-compile for gfx950 (``NVTE_ROCM_ARCH=gfx950``) — no GPU needed —
but it MUST run inside the matching rocm720 base container so the wheel links the same
torch (2.9.1+rocm7.2.0) / Python (3.10) ABI as the runtime image.

Usage (inside a rocm720 base container):
    python build_te_wheel.py --out /out
    python build_te_wheel.py --commit <sha> --out /out          # a different fork commit
    python build_te_wheel.py --repo <url> --branch <name> ...    # override source

After building, upload the .whl to a NEW Release tag (do not overwrite an existing tag)
and point ``WHEELS_TAG_ROCM`` at it; the Dockerfile installs it via the
``transformer_engine-*.whl`` glob, so the exact filename never needs to be hardcoded.
"""

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass, field

TE_REPO_DEFAULT = "https://github.com/XinyuJiangCMU/TransformerEngine.git"
TE_BRANCH_DEFAULT = "miles-dev"
TE_COMMIT_DEFAULT = "619aa3f4"  # miles-dev tip after Zhiyao's fp8 pow2 PR merge

# gfx950 cross-compile env (same as in-container-build.sh / docker/Dockerfile.rocm).
BUILD_ENV = {
    "GPU_ARCH": "gfx950",
    "PYTORCH_ROCM_ARCH": "gfx950",
    "GPU_ARCH_LIST": "gfx950",
    "AMDGPU_TARGET": "gfx950",
    "NVTE_FRAMEWORK": "pytorch",
    "NVTE_ROCM_ARCH": "gfx950",
    "NVTE_USE_HIPBLASLT": "1",
    "NVTE_USE_ROCM": "1",
    "NVTE_NO_LOCAL_VERSION": "1",  # clean PEP440 version (no +<sha> local tag; GitHub mangles '+'->'.')
    "NVTE_FUSED_ATTN": "0",  # build-time scope: skip the fused-attn kernel matrix rebuild
    "CMAKE_PREFIX_PATH": "/opt/rocm:/opt/rocm/hip:/usr/local:/usr",
    "MAX_JOBS": str(os.cpu_count() or 32),
    "PIP_ROOT_USER_ACTION": "ignore",
}


@dataclass
class BuildConfig:
    repo: str = TE_REPO_DEFAULT
    branch: str = TE_BRANCH_DEFAULT
    commit: str = TE_COMMIT_DEFAULT
    env: dict = field(default_factory=lambda: dict(BUILD_ENV))


def _run(cmd, *, env=None, cwd=None):
    merged_env = {**os.environ, **(env or {})}
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}\n", flush=True)
    result = subprocess.run(cmd, env=merged_env, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")


def _ensure_prereqs():
    if shutil.which("apt-get"):
        _run(["apt-get", "update"])
        _run(["apt-get", "install", "-y", "build-essential", "cmake", "git"])
    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is required.")


def build(cfg: BuildConfig, out_dir: str):
    """Clone the fork @ commit, build the wheel, leave it in out_dir."""
    src = "/root/TransformerEngine"
    _ensure_prereqs()
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(src):
        shutil.rmtree(src)

    _run(["git", "clone", "--recursive", "-b", cfg.branch, cfg.repo, src])
    _run(["git", "checkout", cfg.commit], cwd=src)
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=src)
    _run(
        ["pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", out_dir, "-v"],
        env=cfg.env,
        cwd=src,
    )

    wheels = glob.glob(os.path.join(out_dir, "transformer_engine-*.whl"))
    if not wheels:
        raise RuntimeError(f"no transformer_engine wheel produced in {out_dir}")
    for w in wheels:
        print(f"\nBuilt TE wheel: {w}")
    print("\nUpload it to a NEW Release tag, then bump WHEELS_TAG_ROCM in docker/build.py.")


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="/out", help="output directory for the wheel")
    p.add_argument("--repo", default=TE_REPO_DEFAULT, help="TransformerEngine git repository")
    p.add_argument("--branch", default=TE_BRANCH_DEFAULT, help="branch to clone before checkout")
    p.add_argument("--commit", default=TE_COMMIT_DEFAULT, help="commit to build (embedded in the wheel version)")
    args = p.parse_args()

    build(BuildConfig(repo=args.repo, branch=args.branch, commit=args.commit), args.out)


if __name__ == "__main__":
    main()
