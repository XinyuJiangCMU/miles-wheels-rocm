import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import build_apex_wheel as builder


class TestApexWheelBuild(unittest.TestCase):
    def test_accepts_matching_gpu(self):
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                builder.validate_gpu(arch, f"Name: {arch}\nWavefront Size: 64\n")

    def test_rejects_missing_wrong_or_mixed_hardware_and_wrong_wavefront(self):
        for info in ("Name: CPU", "Name: gfx950\nWavefront Size: 64",
                     "Name: gfx942\nName: gfx950\nWavefront Size: 64",
                     "Name: gfx942\nWavefront Size: 32"):
            with self.subTest(info=info), self.assertRaises(RuntimeError):
                builder.validate_gpu("gfx942", info)

    def test_rejects_jit_only_wheel(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "apex.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("fused_weight_gradient_mlp_cuda.py", "# JIT wrapper")
            with self.assertRaisesRegex(RuntimeError, "missing the native"):
                builder.validate_wheel(wheel)

    def test_build_uses_requested_arch_and_records_only_the_new_artifact(self):
        # Mock compilation; exercise command construction, output validation and receipt I/O.
        native = "apex/fused_weight_gradient_mlp_cuda.cpython-312-x86_64-linux-gnu.so"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = argparse.Namespace(gpu_arch="gfx942", out=out, commit=builder.APEX_COMMIT, jobs=8)
            (out / "unrelated.txt").write_text("keep")
            envs = []

            def fake_run(cmd, *, cwd=None, env=None):
                if "--wheel-dir" in cmd:
                    envs.append(env)
                    wheel_dir = Path(cmd[cmd.index("--wheel-dir") + 1])
                    wheel_dir.mkdir()
                    with zipfile.ZipFile(wheel_dir / "apex-1.15.0a0-cp312-cp312-linux_x86_64.whl", "w") as archive:
                        archive.writestr(native, b"fake ELF; no GPU compilation in this test")

            with patch.dict("os.environ", {"GPU_ARCH": "gfx950", "PYTORCH_ROCM_ARCH": "gfx950"}), \
                 patch.object(builder, "check_environment", return_value={"gpu_arch": "gfx942"}), \
                 patch.object(builder, "run", side_effect=fake_run), \
                 patch.object(builder.subprocess, "check_output", return_value=builder.APEX_COMMIT + "\n"):
                builder.build(args)

            self.assertEqual(envs[0]["PYTORCH_ROCM_ARCH"], "gfx942")
            self.assertEqual(envs[0]["GPU_ARCH"], "gfx942")
            self.assertEqual(envs[0]["APEX_BUILD_CPP_OPS"], "1")
            self.assertEqual(envs[0]["APEX_BUILD_CUDA_OPS"], "1")
            self.assertEqual(envs[0]["MAX_JOBS"], "8")
            receipt = json.loads(next(out.glob("*.build.json")).read_text())
            self.assertEqual(receipt["source_commit"], builder.APEX_COMMIT)
            self.assertEqual(receipt["gpu_runtime_validation"], "not run")
            self.assertEqual(receipt["native_ops"], [native])
            self.assertEqual((out / "unrelated.txt").read_text(), "keep")
            self.assertFalse(list(out.glob(".apex-build-*")))

    def test_cli_rejects_bad_arguments_before_touching_build_environment(self):
        script = str(Path(builder.__file__).resolve())
        for extra in (["--gpu-arch", "gfx900"], ["--commit", "main"], ["--jobs", "0"]):
            with self.subTest(extra=extra):
                result = subprocess.run([sys.executable, script, "--gpu-arch", "gfx942",
                                         "--out", "/unused", *extra], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
