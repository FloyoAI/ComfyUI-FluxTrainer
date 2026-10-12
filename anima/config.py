"""Build sd-scripts arguments without importing torch or loading models."""

import copy
import json
import math
import re
from pathlib import Path

SD_SCRIPTS_REVISION = "690ea7f96c23182352ec63def76d431c6120bd2f"
BACKEND_ENV_VERSION = 2
SCRIPTS = {"lora": "anima_train_network.py", "full": "anima_train.py"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
OPTIMIZER_KEYS = {
    "optimizer_type", "optimizer_args", "max_grad_norm", "lr_scheduler",
    "lr_warmup_steps", "lr_scheduler_num_cycles", "lr_scheduler_power",
}


def local_path(value, label, directory=False):
    if not value or not isinstance(value, (str, Path)):
        raise ValueError(f"{label} must be a local path")
    path = Path(value).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{label} does not exist: {path}")
    return str(path)


def positive(value, label, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{label} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def choice(value, choices, label):
    if value not in choices:
        raise ValueError(f"{label} must be one of {', '.join(choices)}")
    return value


def normalize_dataset(dataset, cache_latents, cache_text):
    """Accept the existing FluxTrainer JSON socket; never mutate its contents."""
    alpha_mask = dataset.get("alpha_mask", False)
    payload = dataset.get("datasets")
    data = json.loads(payload) if isinstance(payload, str) else copy.deepcopy(dataset)
    if not isinstance(data, dict) or not isinstance(data.get("datasets"), list) or not data["datasets"]:
        raise ValueError("Dataset must contain at least one dataset")
    # alpha_mask is a FluxTrainer wrapper field, not an sd-scripts top-level field.
    data.pop("alpha_mask", None)
    general = data.get("general", {})
    training_images = 0
    for ds in data["datasets"]:
        settings = {**general, **ds}
        resolution = settings.get("resolution", 1024)
        if isinstance(resolution, int):
            resolution = [resolution, resolution]
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or any(
            not isinstance(n, int) or n < 64 or n % 16 for n in resolution
        ):
            raise ValueError("Anima resolution must have two dimensions divisible by 16 (at least 64)")
        ds["resolution"] = list(resolution)
        for key, default in (("bucket_reso_steps", 64), ("min_bucket_reso", 256), ("max_bucket_reso", 1024)):
            value = settings.get(key, default)
            if not isinstance(value, int) or value <= 0 or value % 16:
                raise ValueError(f"{key} must be a positive multiple of 16")
        if settings.get("min_bucket_reso", 256) > settings.get("max_bucket_reso", 1024):
            raise ValueError("min_bucket_reso must not exceed max_bucket_reso")
        integer(settings.get("batch_size", 1), "batch_size", 1)
        if not ds.get("subsets"):
            raise ValueError("Dataset must contain at least one subset")
        for subset in ds["subsets"]:
            effective = {**settings, **subset}
            integer(effective.get("num_repeats", 1), "num_repeats", 1)
            subset["image_dir"] = local_path(subset.get("image_dir"), "Dataset image directory", directory=True)
            if "metadata_file" in subset:
                subset["metadata_file"] = local_path(subset["metadata_file"], "Dataset metadata")
            if alpha_mask:
                subset["alpha_mask"] = True
            if cache_latents != "disabled" and any(effective.get(k) for k in ("color_aug", "random_crop")):
                raise ValueError("Latent caching requires color_aug and random_crop disabled")
            if cache_text != "disabled" and any(effective.get(k) for k in (
                "shuffle_caption", "token_warmup_step", "caption_tag_dropout_rate", "enable_wildcard",
            )):
                raise ValueError("Text caching requires shuffle_caption, token_warmup_step, caption_tag_dropout_rate and enable_wildcard disabled")
            count = sum(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in Path(subset["image_dir"]).iterdir())
            if not count:
                raise ValueError(f"No supported images directly inside {subset['image_dir']}")
            if not subset.get("is_reg", False):
                training_images += count
    if not training_images:
        raise ValueError("Dataset must contain training images, not only regularization images")
    return data


def build_config(models, dataset, *, training_type="lora", output_name="anima_lora",
                 learning_rate=1e-4, max_train_steps=1500, network_dim=16, network_alpha=16.0,
                 optimizer_type="AdamW", mixed_precision="bf16", save_precision="bf16",
                 cache_latents="disk", cache_text_encoder_outputs="disk",
                 gradient_checkpointing=True, blocks_to_swap=0, save_every_n_steps=250,
                 seed=42, timestep_sampling="sigmoid", discrete_flow_shift=1.0,
                 gradient_accumulation_steps=1, attention_mode="torch", checkpoint_offload="disabled",
                 save_state=False, resume="", network_weights="", optimizer_settings=None,
                 sample_prompts="", sample_every_n_steps=0, t5_tokenizer_path=""):
    choice(training_type, SCRIPTS, "training_type")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", output_name):
        raise ValueError("output_name must be a filename using letters, digits, dots, underscores or hyphens")
    for key, value, choices in (
        ("cache_latents", cache_latents, ("disk", "memory", "disabled")),
        ("cache_text_encoder_outputs", cache_text_encoder_outputs, ("disk", "memory", "disabled")),
        ("mixed_precision", mixed_precision, ("bf16", "fp16", "no")),
        ("save_precision", save_precision, ("bf16", "fp16", "float")),
        ("attention_mode", attention_mode, ("torch", "xformers", "flash")),
        ("checkpoint_offload", checkpoint_offload, ("disabled", "cpu", "unsloth")),
        ("timestep_sampling", timestep_sampling, ("sigma", "uniform", "sigmoid", "shift", "flux_shift")),
    ):
        choice(value, choices, key)
    for key, value in (("learning_rate", learning_rate), ("max_train_steps", max_train_steps),
                       ("gradient_accumulation_steps", gradient_accumulation_steps), ("discrete_flow_shift", discrete_flow_shift)):
        positive(value, key)
    for key, value in (("blocks_to_swap", blocks_to_swap), ("save_every_n_steps", save_every_n_steps),
                       ("sample_every_n_steps", sample_every_n_steps)):
        integer(value, key)
    integer(max_train_steps, "max_train_steps", 1)
    integer(gradient_accumulation_steps, "gradient_accumulation_steps", 1)
    integer(seed, "seed")
    if blocks_to_swap and checkpoint_offload != "disabled":
        raise ValueError("Block swapping cannot be combined with checkpoint offloading")
    if blocks_to_swap > 26:
        raise ValueError("The pinned Anima backend supports at most 26 swapped blocks")
    if checkpoint_offload != "disabled" and not gradient_checkpointing:
        raise ValueError("Checkpoint offloading requires gradient_checkpointing")
    if sample_every_n_steps and not sample_prompts.strip():
        raise ValueError("Sampling requires at least one prompt")
    qwen3 = Path(models.get("qwen3", ""))
    args = {
        "pretrained_model_name_or_path": local_path(models.get("transformer"), "Anima DiT"),
        "qwen3": local_path(models.get("qwen3"), "Qwen3 encoder", directory=qwen3.is_dir()),
        "vae": local_path(models.get("vae"), "Qwen-Image VAE"),
        "output_name": output_name, "save_model_as": "safetensors",
        "learning_rate": learning_rate, "max_train_steps": max_train_steps,
        "optimizer_type": optimizer_type, "lr_scheduler": "constant", "max_grad_norm": 1.0,
        "mixed_precision": mixed_precision, "save_precision": save_precision,
        "seed": seed, "gradient_checkpointing": gradient_checkpointing,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "blocks_to_swap": blocks_to_swap, "attn_mode": attention_mode,
        "split_attn": attention_mode == "xformers", "timestep_sampling": timestep_sampling,
        "discrete_flow_shift": discrete_flow_shift, "weighting_scheme": "none",
        "max_data_loader_n_workers": 0, "vae_batch_size": 1,
        "vae_chunk_size": 64, "vae_disable_cache": True,
        "save_state": save_state, "save_state_on_train_end": save_state,
    }
    for key, mode in (("cache_latents", cache_latents), ("cache_text_encoder_outputs", cache_text_encoder_outputs)):
        args[key] = mode != "disabled"
        args[key + "_to_disk"] = mode == "disk"
    if checkpoint_offload != "disabled":
        args["cpu_offload_checkpointing" if checkpoint_offload == "cpu" else "unsloth_offload_checkpointing"] = True
    if save_every_n_steps:
        args["save_every_n_steps"] = save_every_n_steps
    if sample_every_n_steps:
        args["sample_every_n_steps"] = sample_every_n_steps
    if optimizer_settings:
        settings = copy.deepcopy(optimizer_settings)
        if settings.pop("min_snr_gamma", None) not in (None, 0, 0.0):
            raise ValueError("Anima does not support min_snr_gamma; set it to 0 in OptimizerConfig")
        unknown = settings.keys() - OPTIMIZER_KEYS
        if unknown:
            raise ValueError(f"Unsupported optimizer settings: {', '.join(sorted(unknown))}")
        args.update(settings)
    if args["optimizer_type"].lower() == "adafactor" and not args.get("optimizer_args"):
        args["optimizer_args"] = ["relative_step=False", "scale_parameter=False", "warmup_init=False"]
    if training_type == "lora":
        args.update(network_module="networks.lora_anima", network_train_unet_only=True,
                    network_dim=integer(network_dim, "network_dim", 1), network_alpha=positive(network_alpha, "network_alpha"))
        if network_weights:
            args["network_weights"] = local_path(network_weights, "sd-scripts LoRA weights")
    elif network_weights:
        raise ValueError("network_weights is only available for LoRA training")
    if resume:
        args["resume"] = local_path(resume, "Accelerate resume state", directory=True)
    if t5_tokenizer_path:
        args["t5_tokenizer_path"] = local_path(t5_tokenizer_path, "T5 tokenizer", directory=True)
    return {"training_type": training_type, "args": args,
            "dataset": normalize_dataset(dataset, cache_latents, cache_text_encoder_outputs),
            "sample_prompts": sample_prompts.strip() if sample_every_n_steps else ""}


def to_argv(arguments):
    """Use an argument vector, never a shell command or parse_known_args."""
    result = []
    for key, value in arguments.items():
        if value is None or value is False:
            continue
        if value is True:
            result.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            if value:
                result.append(f"--{key}")
                result.extend(str(item) for item in value)
        else:
            result.append(f"--{key}={value}")
    return result
