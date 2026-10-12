"""ComfyUI-Manager installation hook. Installs dependencies without running training."""

from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    subprocess.run([sys.executable, str(Path(__file__).parent / "tools" / "install_anima_backend.py"),
                    *sys.argv[1:]], check=True)
