"""Backend entry point. Validation returns before constructing a trainer or loading weights."""

import argparse
import importlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile


def check_dit_header(path, blocks_to_swap):
    """Read only safetensors metadata, never tensor data."""
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Anima DiT must be a safetensors checkpoint")
        size = struct.unpack("<Q", raw)[0]
        if size > 100_000_000 or size < 2:
            raise ValueError("Invalid Anima safetensors header")
        header = json.loads(stream.read(size))
    tensors = {key.removeprefix("net.").removeprefix("model.diffusion_model."): value
               for key, value in header.items() if key != "__metadata__"}
    if any(value.get("dtype", "").startswith("F8") for value in tensors.values()):
        raise ValueError("FP8 Anima checkpoints are not supported for training; use bf16/fp16 weights")
    if "llm_adapter.out_proj.weight" not in tensors:
        raise ValueError("Anima checkpoint must include the LLM adapter; separate adapter loading is not implemented by this backend")
    if not 0 <= blocks_to_swap <= 26:
        raise ValueError("The pinned backend supports the 28-block Anima model; blocks_to_swap must be 0–26")


def validate_backend(backend, manifest):
    training_type = manifest["training_type"]
    module_name = {"lora": "anima_train_network", "full": "anima_train"}[training_type]
    module = importlib.import_module(module_name)
    parser = module.setup_parser()
    args = parser.parse_args(manifest["argv"])
    # Strict parsing is deliberate: unknown options must not silently disappear.
    from library import config_util
    from library import args as args_util
    args_util.verify_command_line_training_args(args)
    config_util.BlueprintGenerator(config_util.ConfigSanitizer(True, True, args.masked_loss, True)).generate(
        config_util.load_user_config(args.dataset_config), args
    )
    check_dit_header(args.pretrained_model_name_or_path, args.blocks_to_swap)
    # Loading tokenizer vocabularies is CPU-only and reads no model tensors.
    from library import anima_utils
    anima_utils.load_qwen3_tokenizer(args.qwen3)
    anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)
    import torch
    if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 5):
        raise RuntimeError("Anima requires PyTorch >= 2.5")
    if args.attn_mode == "xformers":
        importlib.import_module("xformers.ops")
    if args.attn_mode == "flash":
        importlib.import_module("flash_attn")
    if "8bit" in args.optimizer_type.lower():
        importlib.import_module("bitsandbytes")
    return module, args


def finalize_checkpoint(args, training_type):
    model_path = Path(args.output_dir) / f"{args.output_name}.safetensors"
    if not model_path.is_file():
        raise RuntimeError(f"Backend completed but final checkpoint is missing: {model_path}")
    if training_type == "lora":
        converter = importlib.import_module("networks.convert_anima_lora_to_comfy")
        # Replace only after conversion succeeds, on the same filesystem.
        with tempfile.TemporaryDirectory(prefix=".anima-export-", dir=model_path.parent) as temporary:
            converted = Path(temporary) / model_path.name
            converter.main(argparse.Namespace(src_path=str(model_path), dst_path=str(converted), reverse=False))
            converted.replace(model_path)
    return model_path


def run(backend, manifest_path, action):
    backend = Path(backend).resolve()
    # All absolute upstream imports (library, networks, train_network) must resolve here.
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    module, args = validate_backend(backend, manifest)
    result = {"action": action, "training_type": manifest["training_type"],
              "model_path": "", "native_model_path": "", "sd_scripts_revision": manifest["sd_scripts_revision"]}
    if action == "validate":
        result["message"] = "Validated arguments, dataset schema, DiT header, tokenizer files and backend imports. No model weights were loaded and no training ran."
    elif action == "train":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Anima training requires a CUDA GPU")
        if args.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bf16; select fp16")
        if manifest["training_type"] == "lora":
            module.AnimaNetworkTrainer().train(args)
        else:
            module.train(args)
        model_path = finalize_checkpoint(args, manifest["training_type"])
        result.update(model_path=str(model_path),
                      native_model_path=str(model_path) if manifest["training_type"] == "full" else "",
                      message="Training completed and checkpoint saved.")
    else:
        raise ValueError(f"Unknown action: {action}")
    result_path = manifest_path.parent / f"{action}_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["message"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--action", choices=("validate", "train"), default="validate")
    args = parser.parse_args()
    run(args.backend, args.manifest, args.action)


if __name__ == "__main__":
    main()
