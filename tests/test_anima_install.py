"""Installation lifecycle and real venv isolation tests; no real Torch import or training."""

from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
import unittest
import venv
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anima import runner
from anima.config import BACKEND_ENV_VERSION, SD_SCRIPTS_REVISION
from tools import install_anima_backend as installer


class InstallationTests(unittest.TestCase):
    def use_patch(self, patcher):
        self.addCleanup(patcher.stop)
        return patcher.start()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="anima copy ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.install_root = self.root / ".anima"
        self.backend = self.install_root / "sd-scripts"
        self.python = self.install_root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.state_file = self.install_root / "install_state.json"
        environment = {key: value for key, value in os.environ.items()
                       if key not in ("FLOYO_ANIMA_PYTHON", "FLOYO_ANIMA_SD_SCRIPTS")}
        self.use_patch(patch.dict(os.environ, environment, clear=True))
        self.use_patch(patch.object(runner, "ROOT", self.root))
        self.git = self.use_patch(patch.object(runner.subprocess, "run", return_value=
            subprocess.CompletedProcess([], 0, stdout=SD_SCRIPTS_REVISION + "\n")))
        self.install = self.use_patch(patch.object(runner, "run_process", side_effect=self.complete_install))

    def complete_install(self, *args, **kwargs):
        self.python.parent.mkdir(parents=True, exist_ok=True)
        self.python.touch()
        self.backend.mkdir(parents=True, exist_ok=True)
        for name in ("anima_train_network.py", "anima_train.py"):
            (self.backend / name).touch()
        self.state_file.write_text(json.dumps({"revision": SD_SCRIPTS_REVISION,
                                               "environment_version": BACKEND_ENV_VERSION,
                                               "comfy_python": os.path.abspath(sys.executable)}))

    def test_fresh_copy_installs_with_comfy_python_and_reuses_completed_setup(self):
        cancel, progress = Mock(), Mock()
        self.assertEqual(runner.backend_paths(check_cancel=cancel, on_line=progress), (self.python, self.backend))
        self.install.assert_called_once_with(
            [sys.executable, "-u", str(self.root / "tools" / "install_anima_backend.py"),
             "--root", str(self.install_root)], cwd=self.root, log_path=self.install_root / "install.log",
            check_cancel=cancel, on_line=progress)
        self.assertTrue(progress.called)
        self.assertTrue(cancel.called)
        self.assertEqual(runner.backend_paths(), (self.python, self.backend))
        self.assertEqual(self.install.call_count, 1)

    def test_partial_corrupt_and_moved_installations_require_setup(self):
        self.complete_install()
        for state in (None, "invalid json", "[]", "{}", json.dumps({"revision": SD_SCRIPTS_REVISION,
                                                    "comfy_python": os.path.abspath(sys.executable)}),
                      json.dumps({"revision": SD_SCRIPTS_REVISION, "environment_version": BACKEND_ENV_VERSION,
                                  "comfy_python": "/another/comfy/python"})):
            with self.subTest(state=state):
                if state is None:
                    self.state_file.unlink()
                else:
                    self.state_file.write_text(state)
                self.install.reset_mock()
                runner.backend_paths()
                self.install.assert_called_once()
        (self.backend / "anima_train.py").unlink()
        self.install.reset_mock()
        runner.backend_paths()
        self.install.assert_called_once()

    def test_failed_setup_propagates_before_creating_a_run_and_can_retry(self):
        self.install.side_effect = RuntimeError("dependency download failed")
        release = Mock(side_effect=AssertionError("must not start training"))
        with patch.object(runner, "prepare_run") as prepare:
            with self.assertRaisesRegex(RuntimeError, "dependency download failed"):
                runner.execute({}, self.root, action="train", before_training=release)
        prepare.assert_not_called()
        release.assert_not_called()
        self.git.assert_not_called()
        self.assertFalse(self.state_file.exists())
        self.install.side_effect = self.complete_install
        self.assertEqual(runner.backend_paths(), (self.python, self.backend))

    def test_incomplete_setup_is_not_accepted_as_success(self):
        self.complete_install()
        self.state_file.write_text("{}")
        self.install.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "setup did not complete.*install.log"):
            runner.backend_paths()
        self.git.assert_not_called()

    def test_cancellation_stops_setup_and_releases_lock(self):
        cancel = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            runner.backend_paths(check_cancel=cancel)
        self.install.assert_not_called()
        runner.backend_paths()
        self.install.assert_called_once()

    def test_concurrent_requests_install_only_once(self):
        entered, waiting, release = threading.Event(), threading.Event(), threading.Event()

        def install(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test installer was not released")
            self.complete_install()

        self.install.side_effect = install
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(runner.backend_paths)
            try:
                self.assertTrue(entered.wait(timeout=3))
                second = pool.submit(runner.backend_paths, check_cancel=waiting.set)
                self.assertTrue(waiting.wait(timeout=3))
            finally:
                release.set()
            self.assertEqual(first.result(timeout=3), (self.python, self.backend))
            self.assertEqual(second.result(timeout=3), (self.python, self.backend))
        self.install.assert_called_once()

    def test_explicit_overrides_are_respected_without_automatic_installation(self):
        with patch.dict(os.environ, {"FLOYO_ANIMA_PYTHON": str(self.python),
                                     "FLOYO_ANIMA_SD_SCRIPTS": str(self.backend)}):
            with self.assertRaisesRegex(RuntimeError, "correct those paths") as error:
                runner.backend_paths()
            self.assertIn(str(self.python), str(error.exception))
            self.complete_install()
            self.state_file.unlink()  # Custom environments do not need a managed marker.
            self.assertEqual(runner.backend_paths(), (self.python, self.backend))
        self.install.assert_not_called()

    def test_manager_hook_uses_current_interpreter_and_forwards_install_options(self):
        with patch.object(sys, "argv", ["install.py", "--root", str(self.install_root)]):
            runpy.run_path(str(ROOT / "install.py"), run_name="__main__")
        self.git.assert_called_once_with(
            [sys.executable, str(ROOT / "tools" / "install_anima_backend.py"),
             "--root", str(self.install_root)], check=True)


class EnvironmentIsolationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="anima isolation ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "comfy-site-packages"
        self.source.mkdir()
        self.add_distribution("torch", "2.8.0", "torch", ["pytorch-triton==3.3.0", "nvidia-runtime==12.8", "filelock"])
        self.add_distribution("torchvision", "0.23.0", "torchvision", ["torch==2.8.0"])
        self.add_distribution("pytorch-triton", "3.3.0", "triton")
        self.add_distribution("nvidia-runtime", "12.8", "nvidia")
        self.add_distribution("peft", "0.17.1", "peft")
        self.add_distribution("triton", "3.1.0", "triton")  # Unrelated/stale metadata in ComfyUI.
        (self.source / "peft" / "__init__.py").write_text("raise RuntimeError('host PEFT must not be imported')\n")
        self.distributions = {dist.metadata["Name"]: dist for dist in metadata.distributions(path=[str(self.source)])}

    def add_distribution(self, name, version, module, requires=()):
        package = self.source / module
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text("VALUE = 'shared from ComfyUI'\n")
        info = self.source / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                                      + "".join(f"Requires-Dist: {entry}\n" for entry in requires))
        (info / "RECORD").write_text(f"{module}/__init__.py,,\n{info.name}/METADATA,,\n{info.name}/RECORD,,\n")

    def selected(self):
        with patch.object(installer.metadata, "distribution", side_effect=self.distributions.__getitem__):
            return installer.torch_distributions()

    def test_selects_only_torch_and_its_required_binary_distributions(self):
        self.assertEqual([dist.metadata["Name"] for dist in self.selected()],
                         ["torch", "torchvision", "pytorch-triton", "nvidia-runtime"])

    def test_repairs_legacy_environment_without_exposing_host_peft(self):
        root = self.root / "backend"
        environment = root / "venv"
        venv.EnvBuilder(system_site_packages=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        site = Path(subprocess.check_output([str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
        legacy = site / "comfyui_environment.pth"
        legacy.write_text(str(self.source) + "\n")
        python, constraints, pins = installer.prepare_environment(root, self.selected())
        self.assertFalse(legacy.exists())
        self.assertIn("include-system-site-packages = false", (environment / "pyvenv.cfg").read_text())
        output = subprocess.check_output([str(python), "-I", "-c", """
import importlib.util, importlib.metadata, json, sys, torch, torchvision, triton, nvidia
assert importlib.util.find_spec('peft') is None
try:
    importlib.metadata.version('peft')
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError('host PEFT metadata leaked')
print(json.dumps({'paths': sys.path, 'torch': torch.VALUE, 'version': importlib.metadata.version('torch')}))
"""], text=True)
        result = json.loads(output)
        self.assertNotIn(str(self.source), result["paths"])
        self.assertEqual(result["torch"], "shared from ComfyUI")
        self.assertEqual(result["version"], "2.8.0")
        self.assertIn("torch==2.8.0", constraints.read_text())
        self.assertNotIn("peft", "\n".join(pins))

    def test_copy_fallback_and_rebuilding_shared_view_preserve_source(self):
        shared = self.root / "shared"
        selected = self.selected()
        before = {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        with patch.object(Path, "symlink_to", side_effect=OSError("symlink permission unavailable")):
            installer.share_torch_packages(shared, selected)
        self.assertFalse((shared / "torch").is_symlink())
        self.assertTrue((shared / "torch" / "__init__.py").is_file())
        installer.share_torch_packages(shared, selected)
        self.assertEqual(before, {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main()
