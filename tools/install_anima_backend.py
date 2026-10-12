"""Install the pinned backend in its own venv. This script never starts training."""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anima.config import BACKEND_ENV_VERSION, SD_SCRIPTS_REVISION


def torch_distributions():
    """Share only ComfyUI's binary Torch stack, not its optional ML packages."""
    names = ["torch", "torchvision"]
    distributions = []
    for name in names:
        dist = metadata.distribution(name)
        distributions.append(dist)
        for requirement in dist.requires or []:
            dependency = re.match(r"[A-Za-z0-9_.-]+", requirement).group().lower().replace("_", "-")
            if dependency.startswith("nvidia-") or dependency in ("triton", "pytorch-triton", "pytorch-triton-rocm"):
                if dependency not in names:
                    try:
                        metadata.distribution(dependency)
                    except metadata.PackageNotFoundError:
                        continue  # CUDA/Triton requirements may target another platform.
                    names.append(dependency)
    if tuple(int(part) for part in distributions[0].version.split(".")[:2]) < (2, 5):
        raise RuntimeError("ComfyUI's Python must have PyTorch >= 2.5 installed")
    return distributions


def share_torch_packages(directory, distributions):
    """Create a dedicated import directory outside the venv so pip cannot alter shared packages."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    pins = []
    sources = {}
    for dist in distributions:
        pins.append(f"{dist.metadata['Name']}=={dist.version}")
        if not dist.files:
            raise RuntimeError(f"Cannot share {dist.metadata['Name']}: installed package file list is missing")
        for top in sorted({path.parts[0] for path in dist.files if not path.is_absolute()}):
            # Exclude scripts outside site-packages and never execute source .pth files.
            if top in ("..", ".") or top.endswith(".pth"):
                continue
            source = Path(dist.locate_file(top)).resolve()
            target = directory / top
            if top in sources:
                if sources[top] != source:
                    raise RuntimeError(f"Conflicting Torch package paths for {top}")
                continue
            sources[top] = source
            if top.endswith((".dist-info", ".egg-info")):
                shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)
            else:
                try:
                    target.symlink_to(source, target_is_directory=source.is_dir())
                except OSError:
                    # Windows may lack symlink permission; copies preserve isolation too.
                    shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)
    return pins


def prepare_environment(root, distributions):
    environment = root / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    target_site = Path(subprocess.check_output([str(python), "-I", "-c",
        "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    # Repair environments created by the old installer, which exposed all ComfyUI packages.
    (target_site / "comfyui_environment.pth").unlink(missing_ok=True)
    shared = root / "torch-packages"
    pins = share_torch_packages(shared, distributions)
    (target_site / "comfyui_torch.pth").write_text(str(shared) + "\n", encoding="utf-8")
    constraints = root / "torch-constraints.txt"
    constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
    return python, constraints, pins


def install():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / ".anima", help="Backend storage directory")
    args = parser.parse_args()
    distributions = torch_distributions()
    root = args.root.expanduser().resolve()
    backend = root / "sd-scripts"
    root.mkdir(parents=True, exist_ok=True)
    state_file = root / "install_state.json"
    # Publish readiness only after dependency and parser import checks succeed.
    state_file.unlink(missing_ok=True)
    if not backend.exists():
        # An interrupted clone must not leave a partial checkout at the final path.
        with tempfile.TemporaryDirectory(prefix=".sd-scripts-download-", dir=root) as staging:
            checkout = Path(staging) / "sd-scripts"
            subprocess.run(["git", "clone", "https://github.com/kohya-ss/sd-scripts.git", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", SD_SCRIPTS_REVISION], check=True)
            checkout.rename(backend)
    else:
        revision = subprocess.check_output(["git", "-C", str(backend), "rev-parse", "HEAD"], text=True).strip()
        if revision != SD_SCRIPTS_REVISION:
            raise RuntimeError(f"Existing backend is at {revision}; expected {SD_SCRIPTS_REVISION}. Choose a new --root.")
    python, constraints, pins = prepare_environment(root, distributions)
    subprocess.run([str(python), "-I", "-m", "pip", "install", "-c", str(constraints),
                    "-r", "requirements.txt", *pins], cwd=backend, check=True)
    subprocess.run([str(python), "-I", "-m", "pip", "check"], check=True)
    subprocess.run([str(python), "-I", "-c",
                    "import torch, torchvision; print('Anima backend PyTorch:', torch.__version__)"], check=True)
    # --help invokes argparse only, never a trainer.
    for script in ("anima_train_network.py", "anima_train.py"):
        subprocess.run([str(python), str(backend / script), "--help"], cwd=backend, check=True, stdout=subprocess.DEVNULL)
    state = {"revision": SD_SCRIPTS_REVISION, "comfy_python": os.path.abspath(sys.executable),
             "environment_version": BACKEND_ENV_VERSION}
    temporary_state = state_file.with_suffix(".tmp")
    temporary_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary_state.replace(state_file)
    print(f"Installed Anima backend {SD_SCRIPTS_REVISION}. No training was run.")
    print(f"FLOYO_ANIMA_SD_SCRIPTS={backend}\nFLOYO_ANIMA_PYTHON={python}")


if __name__ == "__main__":
    install()
