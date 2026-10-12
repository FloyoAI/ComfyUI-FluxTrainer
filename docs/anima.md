# Anima training on Floyo / ComfyUI

This branch adds four nodes under **FluxTrainer / Anima** for Anima DiT LoRA training and full DiT fine-tuning. The Qwen3 encoder remains frozen. Existing Flux, SD3 and SDXL nodes keep their bundled backend. Anima uses upstream sd-scripts in a separate Python process and virtual environment.

## Install

1. Put this repository in ComfyUI's `custom_nodes/ComfyUI-FluxTrainer`. Use it in place of an existing FluxTrainer installation so node IDs are registered once. Install the existing `requirements.txt` with ComfyUI's Python if FluxTrainer is not already installed.
2. Restart ComfyUI. Put your non-FP8 Anima DiT checkpoint in `models/diffusion_models`, Qwen3-0.6B in `models/text_encoders`, and Qwen-Image VAE in `models/vae`. The legacy `unet` and `clip` folder registrations also work. Registered extra model paths are respected.
3. Queue an Anima workflow with **action=validate**. When the node has been copied manually, its first use automatically installs the backend with the Python interpreter running ComfyUI, including when ComfyUI runs in a virtual environment. ComfyUI-Manager can also run this setup through the repository's `install.py` hook.

   Setup downloads sd-scripts at commit `690ea7f96c23182352ec63def76d431c6120bd2f`, creates `.anima/venv`, installs the backend requirements there, and checks both trainers' `--help`. It does not download model weights or start training. It shares only ComfyUI's PyTorch, torchvision, and their CUDA/Triton runtime packages through `.anima/torch-packages`; other dependencies are installed independently. Exact version constraints keep pip from replacing that shared binary stack. PyTorch 2.5 or newer is required. CUDA is required when actually training.

   Setup needs Git, network access, and a writable custom-node directory. Its output streams to the ComfyUI console and is saved in `.anima/install.log`. You can interrupt setup through ComfyUI; a later attempt retries incomplete setup. Successful setup records `.anima/install_state.json`, so subsequent requests reuse the installed backend. First use with `action=train` also installs the backend, then proceeds with the requested training after validation.

   Older backend environments that exposed all of ComfyUI's packages are repaired automatically on first use after updating the node. This removes the inherited-package path that caused the `peft` / Transformers `HybridCache` import error. ComfyUI's packages are not upgraded or uninstalled.

To prepare the backend ahead of time, activate ComfyUI's virtual environment and run this from the custom-node directory:

```sh
python tools/install_anima_backend.py
```

Copy the node's source files between installations; let each ComfyUI installation create its own `.anima` directory. Python virtual environments contain machine-specific paths and should not be copied along with the node.

For a Floyo image build or a shared backend installation, install once with `--root /persistent/anima-backend`, then set these in the ComfyUI server environment:

```sh
export FLOYO_ANIMA_SD_SCRIPTS=/persistent/anima-backend/sd-scripts
export FLOYO_ANIMA_PYTHON=/persistent/anima-backend/venv/bin/python
```

Use the same GPU-compatible ComfyUI Python to create the backend environment. Explicit `FLOYO_ANIMA_*` overrides disable automatic setup; those paths must point to an already prepared backend. Do not install the new sd-scripts requirements into the existing ComfyUI environment: its newer Transformers and Diffusers versions may conflict with the bundled Flux/SDXL backend. Backend installation needs network access; validation and training run offline after setup. Models, tokenizers, and datasets must already exist locally. The backend includes Qwen3 and T5 tokenizer assets; a custom T5 tokenizer directory is optional.

## Workflow

Import either [`anima_lora.json`](../example_workflows/anima_lora.json) or [`anima_full.json`](../example_workflows/anima_full.json). API-format versions are provided alongside them. Both examples default to **validate**. Replace the model filenames and dataset directory before queuing.

```text
Anima Model Select ────┐
                      ├─ Anima Training Config ─ Anima Train / Validate
Anima Dataset Config ─┘
```

The dataset node reads a flat directory of images and matching `.txt` captions. Set `class_tokens` for a fallback caption when a caption file is absent. It enables aspect-ratio buckets. For multiple datasets or regularization images, use the existing FluxTrainer `TrainDatasetGeneralConfig` and `TrainDatasetAdd` nodes instead; their JSON output is accepted.

The configuration node validates paths and settings and creates a plain configuration object. It does not initialize any model. Choose `lora` or `full`; rank and alpha apply only to LoRA. The examples use different illustrative learning rates for the two modes; these are starting values, not validated training recipes.

With `action=validate`, the output node checks the actual upstream argument parser, dataset configuration blueprint, DiT safetensors header, tokenizer files, and backend imports. It returns before constructing a trainer or loading model weights. No caching, forward/backward passes, optimization or sample generation runs. A successful validation cannot establish model tensor compatibility, GPU capacity, or training quality.

When you are ready to train, set `action=train` and queue the workflow. This validates first, unloads ComfyUI's inference models, and starts the trainer. Logs stream to the ComfyUI console, with step progress in the node. ComfyUI interruption terminates the backend process group on Linux, including its subprocesses. Partial checkpoints and logs remain available. Windows terminates the direct backend process; descendant cleanup has only been implemented and tested for Linux.

## Settings and restrictions

- Anima uses a Qwen3-0.6B encoder and Qwen-Image VAE. The pinned loader supports the 28-block Anima architecture and requires the LLM adapter inside the DiT checkpoint. It does not implement the separate `llm_adapter_path` advertised by the upstream guide.
- Use bf16/fp16 checkpoints. FP8 checkpoints and FP8 training switches are unsupported. Choose `torch` attention for the default SDPA implementation. `xformers` automatically enables split attention and requires xformers in the backend environment; `flash` requires flash-attn. SageAttention is not offered for training.
- Resolutions and bucket steps must be divisible by 16. Block swapping accepts 0–26 and cannot be combined with CPU or Unsloth checkpoint offloading. Offloading requires gradient checkpointing.
- Latent caching requires `color_aug` and `random_crop` disabled. Text caching requires caption shuffling, token warmup, tag dropout and wildcards disabled. Anima supports ordinary caption dropout with cached text outputs.
- Disk caches are written beside dataset images by sd-scripts. Mount datasets writable when disk caching is selected. Memory and disabled cache modes are available.
- The optional existing optimizer node can override optimizer and scheduler settings. Set its `min_snr_gamma` to 0; Anima does not use the Stable Diffusion Min-SNR loss. Unsupported dictionary keys are rejected. Default Adafactor arguments disable relative-step behavior so the configured learning rate is used.
- Set sample prompts one per line using upstream syntax, for example `a cat --w 1024 --h 1024 --s 20 --d 42 --l 4`. Sampling remains disabled until `sample_every_n_steps` is positive.
- Enable `save_state` to save Accelerate state at checkpoint intervals and on completion. Set `resume` to that state directory to restore it. Use the same model, dataset, and optimizer settings. `network_weights` instead initializes a LoRA from native sd-scripts weights, such as an intermediate checkpoint; it does not restore optimizer state. The final exported LoRA is in ComfyUI format and is intended for inference.
- These nodes run a complete training job per invocation. They do not connect to Kijai's stepwise `NETWORKTRAINER` loop/save/validation nodes, whose in-process trainer API differs from current sd-scripts.

## Output

Relative output directories are under ComfyUI's configured output directory. Absolute paths can target a Floyo persistent output mount. Each invocation gets a unique directory, including validation calls, so previous output is not overwritten.

The run directory contains `manifest.json` (revision, arguments and exact argument vector), `dataset.json`, an optional saved `workflow.json`, `validation.log`, `status.json`, and a result report. Training also writes `training.log`, checkpoints, and any requested samples/state.

For LoRA, the final `<output_name>.safetensors` is in ComfyUI format. Conversion replaces the native final file only after it succeeds, leaving one final LoRA with the configured name and no `-comfy` suffix. Intermediate checkpoints and saved training state retain their native formats. Full fine-tuning returns the DiT safetensors checkpoint with upstream's ComfyUI-compatible prefix. The node returns the inference model path, run directory and JSON report. Validation returns an empty model path.

`copy_to_model_folder` copies the final inference file into a unique subdirectory of `models/loras/anima` or `models/diffusion_models/anima`. It does not affect validation runs.

## Checks without training

The local suite needs only Python 3.10+ and never imports torch. Point it at the pinned source checkout to exercise the actual upstream parser definitions as well:

```sh
ANIMA_CONTRACT_SOURCE=/path/to/sd-scripts python -m unittest discover -s tests -v
python -m compileall -q anima nodes_anima.py install.py tools tests
```

`tools/check_anima_contract.py` extracts and executes only argument-definition functions from upstream source using Python's AST. It neither imports the training modules nor executes trainer code. It verifies the generated options against both real parsers. Without a source checkout those contract tests are explicitly skipped. CI checks out the pinned source and runs the suite on CPU without training dependencies.

The suite also checks dataset/caching constraints, model folder aliases, node registration, workflow links, manifests, first-use installation, retry and reuse behavior, concurrent setup requests, failure propagation, cancellation, and the validation branch's refusal to invoke a trainer. An isolation regression test creates a real virtual environment with synthetic package metadata and verifies that host PEFT cannot leak into it after repair. Backend `--help` and the workflow's **validate** action provide dependency checks on the deployment machine without training.

For a complete installation check, run this with a Python environment containing ComfyUI and FluxTrainer's dependencies:

```sh
python tools/smoke_anima_install.py --comfyui /path/to/ComfyUI
```

With no backend arguments, this exercises automatic setup under the node's `.anima` directory and ignores environment overrides. Run it from a fresh copy without `.anima` to check installation from scratch. To check an existing custom backend instead, pass both `--backend /path/to/anima-backend/sd-scripts` and `--backend-python /path/to/anima-backend/venv/bin/python`.

This imports the complete node package, then executes both validation workflows through the actual nodes and backend subprocess. It uses temporary images and synthetic safetensors headers with empty tensors. It loads tokenizer vocabularies, but no model weights, and never requests training. It verifies installation and integration rather than checkpoint tensor compatibility.

## Provenance and deployment status

This branch is based on [kijai/ComfyUI-FluxTrainer at `09fef403`](https://github.com/kijai/ComfyUI-FluxTrainer/tree/09fef403d12affdf2f035ac0c39a4668d25ad881), with its upstream Git history, license, and existing trainer code preserved. The Anima backend is [kohya-ss/sd-scripts at `690ea7f9`](https://github.com/kohya-ss/sd-scripts/tree/690ea7f96c23182352ec63def76d431c6120bd2f). It is downloaded at installation rather than mixed into Kijai's modified libraries.

The linked [Kohya issue](https://github.com/bmaltais/kohya_ss/issues/3523) concerns full fine-tuning; this integration also exposes LoRA training through `anima_train_network.py`.

The production [FloyoAI/floyo-musubi](https://github.com/FloyoAI/floyo-musubi/tree/76cef2a518ed493dfe4b469a3fa1ecac6c138b9f) source was reviewed after shell access became available. Its integration uses standard ComfyUI node registration, model-folder lookup, subprocess execution and progress reporting; no additional Floyo service API was found in these node paths. Anima follows those same interfaces and explicitly attaches progress to the executing node ID. One backend difference remains: sd-scripts writes disk caches beside dataset images, while Musubi can use a separate cache directory. Anima therefore needs writable datasets for disk caching.

Verification on Python 3.12 passed all 43 unit/contract tests, the backend installer, both upstream `--help` entry points, package imports registering all 36 nodes, and both real validation workflows with synthetic fixtures. Earlier fresh-copy checks used CPU PyTorch 2.8.0 and ComfyUI `e8665ca8`. The environment-isolation repair was also checked against an existing ComfyUI installation with PyTorch `2.8.0.dev20250325+cu128` and PEFT 0.17.1: backend imports, dependency checks, and both node validation workflows passed with PEFT excluded, and ComfyUI's package inventory was unchanged. The final LoRA export also passed a check using the real upstream converter with synthetic tensors, preserving tensor values and metadata under one output filename. These automated checks loaded no trained model weights and ran no training. A user-run local LoRA training completed successfully; Floyo deployment and full-model training remain unverified.
