"""Isolated backend execution, logs, cancellation, and reproducible run manifests."""

import codecs
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import uuid

from .config import BACKEND_ENV_VERSION, SD_SCRIPTS_REVISION, to_argv

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("worker.py")
BACKEND_SETUP_LOCK = threading.Lock()


def managed_backend_ready(state_file, python, backend):
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(state, dict)
            and state.get("environment_version") == BACKEND_ENV_VERSION
            and state.get("revision") == SD_SCRIPTS_REVISION
            and state.get("comfy_python") == os.path.abspath(sys.executable)
            and python.is_file()
            and all((backend / script).is_file() for script in ("anima_train_network.py", "anima_train.py")))


def backend_paths(*, check_cancel=lambda: None, on_line=lambda line: None):
    backend = Path(os.environ.get("FLOYO_ANIMA_SD_SCRIPTS", ROOT / ".anima" / "sd-scripts")).expanduser().resolve()
    default_python = ROOT / ".anima" / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Do not resolve symlinks: venv/bin/python must retain its venv identity.
    python = Path(os.path.abspath(os.path.expanduser(os.environ.get("FLOYO_ANIMA_PYTHON", str(default_python)))))
    managed = not any(name in os.environ for name in ("FLOYO_ANIMA_SD_SCRIPTS", "FLOYO_ANIMA_PYTHON"))
    if managed:
        # Two queued workflows must not install into the same environment at once.
        while not BACKEND_SETUP_LOCK.acquire(timeout=0.1):
            check_cancel()
        try:
            check_cancel()
            state_file = ROOT / ".anima" / "install_state.json"
            if not managed_backend_ready(state_file, python, backend):
                install_root = ROOT / ".anima"
                install_root.mkdir(parents=True, exist_ok=True)
                on_line("Installing Anima's backend with ComfyUI's Python. This first-time setup downloads dependencies, not model weights.")
                run_process([sys.executable, "-u", str(ROOT / "tools" / "install_anima_backend.py"),
                             "--root", str(install_root)], cwd=ROOT, log_path=install_root / "install.log",
                            check_cancel=check_cancel, on_line=on_line)
                if not managed_backend_ready(state_file, python, backend):
                    raise RuntimeError(f"Anima setup did not complete. See {install_root / 'install.log'}")
        finally:
            BACKEND_SETUP_LOCK.release()
    if not python.is_file() or not (backend / "anima_train_network.py").is_file() or not (backend / "anima_train.py").is_file():
        raise RuntimeError(
            f"Anima backend paths are incomplete. Python: {python}; sd-scripts: {backend}. "
            "If you configured FLOYO_ANIMA_PYTHON or FLOYO_ANIMA_SD_SCRIPTS, correct those paths "
            "or remove both overrides to use automatic setup."
        )
    revision = subprocess.run(["git", "-C", str(backend), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if revision != SD_SCRIPTS_REVISION:
        raise RuntimeError(f"Anima requires sd-scripts {SD_SCRIPTS_REVISION}; found {revision}")
    return python, backend


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_run(config, output_root, workflow=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root).resolve() / f"{config['args']['output_name']}-{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = run_dir / "dataset.json"
    write_json(dataset_path, config["dataset"])
    args = {**config["args"], "dataset_config": str(dataset_path), "output_dir": str(run_dir)}
    if config.get("sample_prompts"):
        prompt_path = run_dir / "sample_prompts.txt"
        prompt_path.write_text(config["sample_prompts"] + "\n", encoding="utf-8")
        args["sample_prompts"] = str(prompt_path)
    manifest = {"sd_scripts_revision": SD_SCRIPTS_REVISION, "training_type": config["training_type"],
                "arguments": args, "argv": to_argv(args)}
    write_json(run_dir / "manifest.json", manifest)
    if workflow is not None:
        write_json(run_dir / "workflow.json", workflow)
    return run_dir


def stop_process(process):
    """Terminate the whole POSIX process group, then reap the child."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)
    if os.name == "posix":
        # A child can outlive its parent and ignore SIGTERM. Reap its process group too.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_process(command, *, cwd, log_path, check_cancel=lambda: None, on_line=lambda line: None):
    """Read pipes on a daemon thread so cancellation works during silent operations."""
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # Prevent ComfyUI/custom-node modules leaking into the external backend.
    env.pop("PYTHONPATH", None)
    check_cancel()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=os.name == "posix")
    messages = queue.Queue(maxsize=256)
    reader_stop = threading.Event()
    tail = deque(maxlen=40)

    def put(value):
        while not reader_stop.is_set():
            try:
                messages.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def read_output():
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""
        try:
            while chunk := process.stdout.read1(4096):
                pending += decoder.decode(chunk)
                parts = re.split(r"[\r\n]", pending)
                pending = parts.pop()
                for line in parts:
                    if line:
                        put(line)
            pending += decoder.decode(b"", final=True)
            if pending:
                put(pending)
        finally:
            put(None)

    thread = threading.Thread(target=read_output, daemon=True)
    thread.start()
    try:
        with Path(log_path).open("w", encoding="utf-8") as log:
            finished = False
            while not finished:
                check_cancel()
                try:
                    line = messages.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    finished = True
                else:
                    tail.append(line)
                    log.write(line + "\n")
                    log.flush()
                    on_line(line)
            while process.poll() is None:
                check_cancel()
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        if process.returncode:
            raise RuntimeError(f"Anima backend exited with code {process.returncode}. Log: {log_path}\n" + "\n".join(tail))
    finally:
        reader_stop.set()
        stop_process(process)
        thread.join(timeout=2)
        process.stdout.close()


def execute(config, output_root, *, action="validate", workflow=None, check_cancel=lambda: None,
            on_line=lambda line: None, before_training=lambda: None):
    if action not in ("validate", "train"):
        raise ValueError("action must be validate or train")
    python, backend = backend_paths(check_cancel=check_cancel, on_line=on_line)
    run_dir = prepare_run(config, output_root, workflow)
    base_command = [str(python), "-u", str(WORKER), "--backend", str(backend),
                    "--manifest", str(run_dir / "manifest.json")]
    write_json(run_dir / "status.json", {"status": "validating", "action": action})
    try:
        run_process(base_command + ["--action", "validate"], cwd=backend, log_path=run_dir / "validation.log",
                    check_cancel=check_cancel, on_line=on_line)
        if action == "train":
            check_cancel()
            before_training()
            write_json(run_dir / "status.json", {"status": "training", "action": action})
            run_process(base_command + ["--action", "train"], cwd=backend, log_path=run_dir / "training.log",
                        check_cancel=check_cancel, on_line=on_line)
        result = json.loads((run_dir / f"{action}_result.json").read_text(encoding="utf-8"))
        write_json(run_dir / "status.json", {"status": "validated" if action == "validate" else "completed", "action": action})
        return result, run_dir
    except BaseException as exc:
        write_json(run_dir / "status.json", {"status": "interrupted" if type(exc).__name__ in (
            "InterruptProcessingException", "KeyboardInterrupt") else "failed", "error": str(exc), "action": action})
        raise
