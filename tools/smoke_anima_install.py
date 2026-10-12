"""Validate real ComfyUI/backend imports with synthetic files; never load weights or train."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--backend", type=Path, help="Omit both backend options to exercise automatic first-use setup")
    parser.add_argument("--backend-python", type=Path)
    options = parser.parse_args()
    if bool(options.backend) != bool(options.backend_python):
        parser.error("Provide both --backend and --backend-python, or omit both for automatic setup")
    # ComfyUI still runs in CPU mode below. Let installed GPU extensions detect
    # their driver during import; hiding it breaks some Triton versions.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if options.backend:
        os.environ.update(FLOYO_ANIMA_SD_SCRIPTS=str(options.backend.resolve()),
                          FLOYO_ANIMA_PYTHON=str(options.backend_python.absolute()))
    else:
        os.environ.pop("FLOYO_ANIMA_SD_SCRIPTS", None)
        os.environ.pop("FLOYO_ANIMA_PYTHON", None)
    sys.path.insert(0, str(options.comfyui.resolve()))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import folder_paths
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("fluxtrainer_smoke", root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    mappings = package.NODE_CLASS_MAPPINGS
    print(f"Registered {len(mappings)} nodes with real ComfyUI imports.")

    with tempfile.TemporaryDirectory(prefix="anima-validation-") as temporary:
        directory = Path(temporary)
        images = directory / "images"
        images.mkdir()
        Image.new("RGB", (64, 64)).save(images / "example.png")
        (images / "example.txt").write_text("a synthetic validation image", encoding="utf-8")
        # Empty tensors create valid safetensors headers without model weights.
        header = json.dumps({"llm_adapter.out_proj.weight": {
            "dtype": "BF16", "shape": [0], "data_offsets": [0, 0],
        }}).encode()
        header += b" " * (-len(header) % 8)
        for category, filename in (("diffusion_models", "anima-smoke.safetensors"),
                                   ("text_encoders", "qwen-smoke.safetensors"), ("vae", "vae-smoke.safetensors")):
            (directory / filename).write_bytes(struct.pack("<Q", len(header)) + header)
            folder_paths.add_model_folder_path(category, str(directory))
        models = mappings["AnimaModelSelect"]().select(
            "anima-smoke.safetensors", "qwen-smoke.safetensors", "vae-smoke.safetensors",
        )[0]
        dataset = mappings["AnimaDatasetConfig"]().create_config(str(images), 64, 1, 1, ".txt", "")[0]
        for training_type in ("lora", "full"):
            config = mappings["InitAnimaTraining"]().configure(models, dataset, training_type=training_type)[0]
            output = mappings["AnimaTrain"]().run(config, action="validate", output_dir=str(directory / training_type),
                                                 unique_id="anima-smoke")
            model_path, run_directory, report = output["result"]
            assert model_path == ""
            assert json.loads(report)["action"] == "validate"
            status = json.loads((Path(run_directory) / "status.json").read_text())
            assert status["status"] == "validated"
            assert not (Path(run_directory) / "training.log").exists()
            print(f"{training_type}: real node → worker → upstream parser/blueprint/tokenizer validation passed.")
    print("Synthetic fixtures removed. No model weights were loaded, and no training ran.")


if __name__ == "__main__":
    main()
