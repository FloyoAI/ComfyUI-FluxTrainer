"""Anima nodes backed by an isolated, pinned sd-scripts installation."""

import json
import logging
from pathlib import Path
import re
import shutil

import folder_paths

from .anima.config import build_config, local_path
from .anima.runner import execute

log = logging.getLogger(__name__)
CATEGORY = "FluxTrainer/Anima"


def model_names(*categories):
    names = set()
    for category in categories:
        if category in folder_paths.folder_names_and_paths:
            names.update(folder_paths.get_filename_list(category))
    return sorted(names)


def model_path(name, *categories):
    for category in categories:
        if category in folder_paths.folder_names_and_paths:
            path = folder_paths.get_full_path(category, name)
            if path:
                return local_path(path, category)
    raise ValueError(f"Model not found in {', '.join(categories)}: {name}")


class AnimaModelSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "transformer": (model_names("diffusion_models", "unet"), {"tooltip": "Anima bf16/fp16 DiT with its LLM adapter. FP8 is unsupported."}),
            "qwen3": (model_names("text_encoders", "clip"), {"tooltip": "Qwen3-0.6B text encoder"}),
            "vae": (model_names("vae"), {"tooltip": "Qwen-Image VAE"}),
        }}

    RETURN_TYPES = ("TRAIN_ANIMA_MODELS",)
    RETURN_NAMES = ("anima_models",)
    FUNCTION = "select"
    CATEGORY = CATEGORY

    def select(self, transformer, qwen3, vae):
        return ({"transformer": model_path(transformer, "diffusion_models", "unet"),
                 "qwen3": model_path(qwen3, "text_encoders", "clip"),
                 "vae": model_path(vae, "vae")},)


class AnimaDatasetConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_dir": ("STRING", {"default": "", "tooltip": "Folder containing images and matching caption files; relative to ComfyUI's working directory."}),
            "resolution": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 256}),
            "num_repeats": ("INT", {"default": 1, "min": 1, "max": 1000}),
            "caption_extension": ("STRING", {"default": ".txt"}),
            "class_tokens": ("STRING", {"default": "", "tooltip": "Fallback caption for images without a caption file."}),
        }}

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "create_config"
    CATEGORY = CATEGORY

    def create_config(self, image_dir, resolution, batch_size, num_repeats, caption_extension, class_tokens):
        if not caption_extension.startswith(".") or "/" in caption_extension or "\\" in caption_extension:
            raise ValueError("caption_extension must be a file extension, such as .txt")
        dataset = {"general": {"caption_extension": caption_extension}, "datasets": [{
            "resolution": [resolution, resolution], "batch_size": batch_size,
            "enable_bucket": True, "bucket_no_upscale": True, "bucket_reso_steps": 16,
            "min_bucket_reso": 256, "max_bucket_reso": max(1024, resolution),
            "subsets": [{"image_dir": local_path(image_dir, "image_dir", directory=True),
                         "num_repeats": num_repeats, "class_tokens": class_tokens}],
        }]}
        return ({"datasets": json.dumps(dataset), "alpha_mask": False},)


class InitAnimaTraining:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "anima_models": ("TRAIN_ANIMA_MODELS",),
            "dataset": ("JSON", {"tooltip": "AnimaDatasetConfig or FluxTrainer TrainDatasetAdd output."}),
            "training_type": (["lora", "full"], {"default": "lora"}),
            "output_name": ("STRING", {"default": "anima_lora"}),
            "learning_rate": ("FLOAT", {"default": 0.0001, "min": 0.00000001, "max": 1.0, "step": 0.000001}),
            "max_train_steps": ("INT", {"default": 1500, "min": 1, "max": 1000000}),
            "network_dim": ("INT", {"default": 16, "min": 1, "max": 1024, "tooltip": "LoRA rank; ignored for full fine-tuning."}),
            "network_alpha": ("FLOAT", {"default": 16.0, "min": 0.01, "max": 1024.0, "tooltip": "LoRA alpha; ignored for full fine-tuning."}),
            "optimizer_type": (["AdamW", "AdamW8bit", "Adafactor"], {"default": "AdamW"}),
            "mixed_precision": (["bf16", "fp16", "no"], {"default": "bf16"}),
            "save_precision": (["bf16", "fp16", "float"], {"default": "bf16"}),
            "cache_latents": (["disk", "memory", "disabled"], {"default": "disk"}),
            "cache_text_encoder_outputs": (["disk", "memory", "disabled"], {"default": "disk"}),
            "gradient_checkpointing": ("BOOLEAN", {"default": True}),
            "blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 26}),
            "save_every_n_steps": ("INT", {"default": 250, "min": 0, "max": 1000000, "tooltip": "0 saves only the final model."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295}),
        }, "optional": {
            "optimizer_settings": ("ARGS", {"tooltip": "FluxTrainer optimizer settings; set min_snr_gamma to 0."}),
            "gradient_accumulation_steps": ("INT", {"default": 1, "min": 1, "max": 1024}),
            "timestep_sampling": (["sigmoid", "sigma", "uniform", "shift", "flux_shift"], {"default": "sigmoid"}),
            "discrete_flow_shift": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0}),
            "attention_mode": (["torch", "xformers", "flash"], {"default": "torch"}),
            "checkpoint_offload": (["disabled", "cpu", "unsloth"], {"default": "disabled"}),
            "save_state": ("BOOLEAN", {"default": False, "tooltip": "Save Accelerate state for resuming (uses additional disk space)."}),
            "resume": ("STRING", {"default": "", "tooltip": "Local Accelerate state directory."}),
            "network_weights": ("STRING", {"default": "", "tooltip": "Initialize LoRA weights from a native sd-scripts checkpoint. The final exported LoRA is in ComfyUI format."}),
            "sample_prompts": ("STRING", {"default": "", "multiline": True, "tooltip": "One sd-scripts sample prompt per line; supports --w, --h, --s, --d, --l and --n."}),
            "sample_every_n_steps": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            "t5_tokenizer_path": ("STRING", {"default": "", "tooltip": "Optional local tokenizer directory. Bundled tokenizer is used by default."}),
        }}

    RETURN_TYPES = ("ANIMA_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = CATEGORY

    def configure(self, anima_models, dataset, **settings):
        return (build_config(anima_models, dataset, **settings),)


class AnimaTrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("ANIMA_CONFIG",),
            "action": (["validate", "train"], {"default": "validate", "tooltip": "Validate checks configuration and backend dependencies without loading weights or running training."}),
            "output_dir": ("STRING", {"default": "anima", "tooltip": "Relative paths are inside ComfyUI output. Each invocation creates its own run folder."}),
            "copy_to_model_folder": ("BOOLEAN", {"default": False}),
        }, "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_path", "run_directory", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Training and file validation are side effects; never reuse a cached result.
        return float("nan")

    def run(self, config, action="validate", output_dir="anima", copy_to_model_folder=False, extra_pnginfo=None, unique_id=None):
        import comfy.model_management as mm
        import comfy.utils

        root = Path(output_dir).expanduser()
        if not root.is_absolute():
            root = Path(folder_paths.get_output_directory()) / root
        pbar = comfy.utils.ProgressBar(config["args"]["max_train_steps"], node_id=unique_id)

        def on_line(line):
            log.info("[Anima] %s", line)
            if "steps:" in line:
                match = re.search(r"\|\s*(\d+)/(\d+)", line)
                if match:
                    pbar.update_absolute(int(match[1]), int(match[2]))

        def release_models():
            mm.unload_all_models()
            mm.soft_empty_cache()

        result, run_dir = execute(config, root, action=action,
                                  workflow=(extra_pnginfo or {}).get("workflow"),
                                  check_cancel=mm.throw_exception_if_processing_interrupted,
                                  on_line=on_line, before_training=release_models)
        path = result["model_path"]
        if action == "train" and copy_to_model_folder:
            category = "loras" if config["training_type"] == "lora" else "diffusion_models"
            destination = Path(folder_paths.models_dir) / category / "anima" / run_dir.name
            destination.mkdir(parents=True, exist_ok=False)
            path = str(destination / Path(path).name)
            shutil.copy2(result["model_path"], path)
        report = json.dumps({**result, "model_path": path, "run_directory": str(run_dir)}, indent=2)
        return {"ui": {"text": [report]}, "result": (path, str(run_dir), report)}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    AnimaModelSelect, AnimaDatasetConfig, InitAnimaTraining, AnimaTrain,
)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaModelSelect": "Anima Model Select",
    "AnimaDatasetConfig": "Anima Dataset Config",
    "InitAnimaTraining": "Anima Training Config",
    "AnimaTrain": "Anima Train / Validate",
}
