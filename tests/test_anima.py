"""CPU-only integration tests. No torch import, model download, or training call."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima.config import build_config, to_argv, SD_SCRIPTS_REVISION
from anima import runner, worker
from tools.check_anima_contract import upstream_parsers


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="anima test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images with spaces"
        self.images.mkdir()
        (self.images / "example.png").write_bytes(b"test fixture, never decoded")
        self.models = {}
        for name in ("transformer", "qwen3", "vae"):
            path = self.root / f"{name}.safetensors"
            path.touch()
            self.models[name] = str(path)
        self.dataset = {"datasets": json.dumps({"general": {"caption_extension": ".txt"}, "datasets": [{
            "resolution": [1024, 1024], "batch_size": 1, "enable_bucket": True,
            "subsets": [{"image_dir": str(self.images), "num_repeats": 1}],
        }]}), "alpha_mask": False}

    def config(self, **kwargs):
        return build_config(self.models, self.dataset, **kwargs)

    def dataset_options(self, **options):
        data = json.loads(self.dataset["datasets"])
        data["general"].update(options)
        self.dataset["datasets"] = json.dumps(data)


class ConfigTests(Fixture):
    def test_lora_flags_and_full_mode_separation(self):
        lora = self.config()["args"]
        full = self.config(training_type="full")["args"]
        self.assertEqual(lora["network_module"], "networks.lora_anima")
        self.assertTrue(lora["network_train_unet_only"])
        self.assertFalse(any(key.startswith("network_") for key in full))
        self.assertNotIn("fp8_base", lora)
        self.assertEqual(lora["attn_mode"], "torch")

    def test_dataset_is_not_mutated_and_mask_reaches_subsets(self):
        self.dataset["alpha_mask"] = True
        original = copy.deepcopy(self.dataset)
        config = self.config()
        self.assertEqual(self.dataset, original)
        self.assertTrue(config["dataset"]["datasets"][0]["subsets"][0]["alpha_mask"])
        self.assertNotIn("alpha_mask", config["dataset"])

    def test_cache_modes(self):
        for mode in ("disk", "memory", "disabled"):
            with self.subTest(mode=mode):
                args = self.config(cache_latents=mode, cache_text_encoder_outputs=mode)["args"]
                for key in ("cache_latents", "cache_text_encoder_outputs"):
                    self.assertEqual(args[key], mode != "disabled")
                    self.assertEqual(args[key + "_to_disk"], mode == "disk")

    def test_caption_cache_incompatibilities(self):
        for key in ("shuffle_caption", "token_warmup_step", "caption_tag_dropout_rate", "enable_wildcard"):
            with self.subTest(key=key):
                self.dataset_options(**{key: 1})
                with self.assertRaisesRegex(ValueError, "Text caching"):
                    self.config()
                self.config(cache_text_encoder_outputs="disabled")
                self.dataset_options(**{key: 0})

    def test_caption_dropout_is_supported_with_cache(self):
        self.dataset_options(caption_dropout_rate=0.1)
        self.config()

    def test_latent_cache_incompatibilities(self):
        self.dataset_options(color_aug=True)
        with self.assertRaisesRegex(ValueError, "Latent caching"):
            self.config()
        self.config(cache_latents="disabled")

    def test_bad_resolution_empty_dataset_missing_files_and_bad_name(self):
        for resolution in ([1000, 1024], [0, 1024], [1024]):
            data = json.loads(self.dataset["datasets"])
            data["datasets"][0]["resolution"] = resolution
            with self.assertRaisesRegex(ValueError, "resolution"):
                build_config(self.models, data)
        with self.assertRaisesRegex(ValueError, "filename"):
            self.config(output_name="../escape")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.config(learning_rate=float("nan"))
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.config(network_weights="does-not-exist")
        (self.images / "example.png").unlink()
        with self.assertRaisesRegex(ValueError, "No supported images"):
            self.config()

    def test_memory_option_conflicts(self):
        with self.assertRaisesRegex(ValueError, "Block swapping"):
            self.config(blocks_to_swap=10, checkpoint_offload="cpu")
        with self.assertRaisesRegex(ValueError, "requires gradient"):
            self.config(gradient_checkpointing=False, checkpoint_offload="unsloth")
        args = self.config(attention_mode="xformers")["args"]
        self.assertTrue(args["split_attn"])
        with self.assertRaises(ValueError):
            self.config(attention_mode="sageattn")
        with self.assertRaises(ValueError):
            self.config(blocks_to_swap=27)
        with self.assertRaises(ValueError):
            self.config(max_train_steps=1.5)

    def test_optimizer_settings_cannot_override_paths_or_enable_sd_loss(self):
        with self.assertRaisesRegex(ValueError, "min_snr_gamma"):
            self.config(optimizer_settings={"min_snr_gamma": 5})
        with self.assertRaisesRegex(ValueError, "Unsupported optimizer"):
            self.config(optimizer_settings={"output_dir": "/tmp/unexpected"})
        self.assertEqual(self.config(optimizer_settings={"min_snr_gamma": None, "optimizer_type": "AdamW8bit"})["args"]["optimizer_type"], "AdamW8bit")

    def test_argv_preserves_paths_and_omits_false(self):
        argv = to_argv({"qwen3": "/a path/'quoted'/model", "cache_latents": True,
                        "fp8_base": False, "resume": None, "optimizer_args": ["betas=(0.9, 0.99)"]})
        self.assertEqual(argv, ["--qwen3=/a path/'quoted'/model", "--cache_latents", "--optimizer_args", "betas=(0.9, 0.99)"])

    def test_run_manifest_is_unique_and_preserves_settings(self):
        config = self.config(sample_prompts="a cat --w 1024 --h 1024", sample_every_n_steps=50)
        first = runner.prepare_run(config, self.root, {"nodes": []})
        second = runner.prepare_run(config, self.root)
        self.assertNotEqual(first, second)
        manifest = json.loads((first / "manifest.json").read_text())
        self.assertEqual(manifest["sd_scripts_revision"], SD_SCRIPTS_REVISION)
        self.assertEqual(Path(manifest["arguments"]["dataset_config"]).parent, first)
        self.assertEqual(Path(manifest["arguments"]["sample_prompts"]).read_text(), "a cat --w 1024 --h 1024\n")
        self.assertEqual(json.loads((first / "workflow.json").read_text()), {"nodes": []})
        self.assertNotIn("output_dir", config["args"])


class HeaderTests(Fixture):
    def header(self, dtype="BF16", adapter=True):
        data = {"net.blocks.0.weight": {"dtype": dtype}}
        if adapter:
            data["net.llm_adapter.out_proj.weight"] = {"dtype": dtype}
        raw = json.dumps(data).encode()
        path = Path(self.models["transformer"])
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        return path

    def test_metadata_only_check(self):
        worker.check_dit_header(self.header(), 26)

    def test_rejects_fp8_missing_adapter_and_too_many_blocks(self):
        for path, count, message in ((self.header("F8_E4M3"), 0, "FP8"),):
            with self.assertRaisesRegex(ValueError, message):
                worker.check_dit_header(path, count)
        with self.assertRaisesRegex(ValueError, "LLM adapter"):
            worker.check_dit_header(self.header(adapter=False), 0)
        with self.assertRaisesRegex(ValueError, "0–26"):
            worker.check_dit_header(self.header(), 27)


class ExportTests(Fixture):
    def setUp(self):
        super().setUp()
        self.args = argparse.Namespace(output_dir=str(self.root), output_name="anima_lora")
        self.checkpoint = self.root / "anima_lora.safetensors"
        self.checkpoint.write_bytes(b"native checkpoint")

    def test_lora_export_replaces_native_file_and_keeps_original_name(self):
        def convert(args):
            self.assertEqual(Path(args.src_path), self.checkpoint)
            self.assertEqual(self.checkpoint.read_bytes(), b"native checkpoint")
            self.assertFalse(args.reverse)
            Path(args.dst_path).write_bytes(b"comfy checkpoint")

        with patch.object(worker.importlib, "import_module", return_value=types.SimpleNamespace(main=convert)):
            result = worker.finalize_checkpoint(self.args, "lora")
        self.assertEqual(result, self.checkpoint)
        self.assertEqual(result.read_bytes(), b"comfy checkpoint")
        self.assertFalse((self.root / "anima_lora-comfy.safetensors").exists())
        self.assertEqual(list(self.root.glob(".anima-export-*")), [])

    def test_failed_conversion_preserves_native_checkpoint_and_removes_partial_export(self):
        def convert(args):
            Path(args.dst_path).write_bytes(b"incomplete export")
            raise RuntimeError("conversion failed")

        with patch.object(worker.importlib, "import_module", return_value=types.SimpleNamespace(main=convert)):
            with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                worker.finalize_checkpoint(self.args, "lora")
        self.assertEqual(self.checkpoint.read_bytes(), b"native checkpoint")
        self.assertEqual(list(self.root.glob(".anima-export-*")), [])

    def test_full_model_export_does_not_convert_checkpoint(self):
        with patch.object(worker.importlib, "import_module") as importer:
            self.assertEqual(worker.finalize_checkpoint(self.args, "full"), self.checkpoint)
        importer.assert_not_called()
        self.assertEqual(self.checkpoint.read_bytes(), b"native checkpoint")

    def test_missing_final_checkpoint_fails_before_conversion(self):
        self.checkpoint.unlink()
        with patch.object(worker.importlib, "import_module") as importer:
            with self.assertRaisesRegex(RuntimeError, "final checkpoint is missing"):
                worker.finalize_checkpoint(self.args, "lora")
        importer.assert_not_called()


class ProcessTests(Fixture):
    @unittest.skipUnless(os.name == "posix", "Process groups are a POSIX feature")
    def test_cancellation_kills_descendant_that_ignores_sigterm(self):
        child_pid_file = self.root / "descendant.pid"
        # A harmless sleeping descendant; this exercises the same process-tree cleanup.
        code = ("import subprocess, sys, time; "
                "subprocess.Popen([sys.executable, '-c', "
                "\"import signal,time,os; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "Path(os.environ['ANIMA_TEST_PID']).write_text(str(os.getpid())); time.sleep(20)\"]); "
                "time.sleep(20)")

        def cancel():
            if child_pid_file.exists():
                raise KeyboardInterrupt("cancelled subprocess tree")

        with patch.dict(os.environ, {"ANIMA_TEST_PID": str(child_pid_file)}):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_process([sys.executable, "-c", code], cwd=self.root,
                                   log_path=self.root / "tree.log", check_cancel=cancel)
        pid = int(child_pid_file.read_text())
        # A reparented Linux child may briefly remain a zombie until init reaps it.
        stat = Path(f"/proc/{pid}/stat")
        deadline = time.monotonic() + 2
        while stat.exists() and stat.read_text().split()[2] != "Z" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(not stat.exists() or stat.read_text().split()[2] == "Z")

    def test_preflight_failure_never_requests_training(self):
        before = Mock(side_effect=AssertionError("training forbidden"))
        with patch.object(runner, "backend_paths", return_value=(Path(sys.executable), self.root)), \
             patch.object(runner, "run_process", side_effect=RuntimeError("invalid dataset")) as process:
            with self.assertRaisesRegex(RuntimeError, "invalid dataset"):
                runner.execute(self.config(), self.root, action="train", before_training=before)
        before.assert_not_called()
        self.assertEqual(process.call_count, 1)
        status_path = next(self.root.glob("anima_lora-*/status.json"))
        self.assertEqual(json.loads(status_path.read_text())["status"], "failed")

    def test_logs_and_carriage_return_progress(self):
        lines = []
        runner.run_process([sys.executable, "-c", "print('first\\rsecond\\n日本語', flush=True)"],
                           cwd=self.root, log_path=self.root / "run.log", on_line=lines.append)
        self.assertEqual(lines, ["first", "second", "日本語"])

    def test_error_propagates_with_log(self):
        with self.assertRaisesRegex(RuntimeError, "code 7.*Log:"):
            runner.run_process([sys.executable, "-c", "print('specific failure'); raise SystemExit(7)"],
                               cwd=self.root, log_path=self.root / "run.log")
        self.assertIn("specific failure", (self.root / "run.log").read_text())

    def test_silent_child_is_cancelled_and_reaped(self):
        started = time.monotonic()
        captured = []
        real_popen = subprocess.Popen

        def popen(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            captured.append(process)
            return process

        def cancel():
            if time.monotonic() - started > 0.25:
                raise KeyboardInterrupt("cancelled test sleeper")

        with patch.object(runner.subprocess, "Popen", side_effect=popen):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_process([sys.executable, "-c", "import time; time.sleep(20)"],
                                   cwd=self.root, log_path=self.root / "cancel.log", check_cancel=cancel)
        self.assertLess(time.monotonic() - started, 3)
        self.assertIsNotNone(captured[0].poll())

    def test_execute_validate_never_releases_models_or_requests_training(self):
        actions = []

        def harmless_backend(command, **kwargs):
            action = command[-1]
            actions.append(action)
            manifest = Path(command[command.index("--manifest") + 1])
            runner.write_json(manifest.parent / "validate_result.json", {"model_path": "", "action": "validate"})

        before = Mock(side_effect=AssertionError("Training must never start"))
        with patch.object(runner, "backend_paths", return_value=(Path(sys.executable), self.root)), \
             patch.object(runner, "run_process", side_effect=harmless_backend):
            result, run_dir = runner.execute(self.config(), self.root, before_training=before)
        self.assertEqual(actions, ["validate"])
        before.assert_not_called()
        self.assertEqual(result["model_path"], "")
        self.assertEqual(json.loads((run_dir / "status.json").read_text())["status"], "validated")

    def test_worker_validate_returns_before_any_trainer_call(self):
        run_dir = runner.prepare_run(self.config(), self.root)
        module = types.SimpleNamespace(AnimaNetworkTrainer=Mock(side_effect=AssertionError("training forbidden")),
                                       train=Mock(side_effect=AssertionError("training forbidden")))
        old_cwd, old_path = os.getcwd(), sys.path[:]
        try:
            with patch.object(worker, "validate_backend", return_value=(module, argparse.Namespace())):
                result = worker.run(self.root, run_dir / "manifest.json", "validate")
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path
        self.assertEqual(result["action"], "validate")
        module.train.assert_not_called()
        module.AnimaNetworkTrainer.assert_not_called()


class UpstreamContractTests(Fixture):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get("ANIMA_CONTRACT_SOURCE", str(ROOT / ".anima" / "sd-scripts"))
        if not (Path(source) / "anima_train.py").is_file():
            raise unittest.SkipTest("Set ANIMA_CONTRACT_SOURCE to the pinned sd-scripts source to check actual parsers")
        cls.parsers = upstream_parsers(source)

    def test_all_modes_caches_and_precision_pass_actual_upstream_parsers(self):
        for mode in ("lora", "full"):
            for cache in ("disk", "memory", "disabled"):
                for precision in ("bf16", "fp16", "no"):
                    with self.subTest(mode=mode, cache=cache, precision=precision):
                        config = self.config(training_type=mode, mixed_precision=precision,
                                             cache_latents=cache, cache_text_encoder_outputs=cache)
                        run_dir = runner.prepare_run(config, self.root)
                        manifest = json.loads((run_dir / "manifest.json").read_text())
                        args = self.parsers[mode].parse_args(manifest["argv"])
                        self.assertEqual(args.pretrained_model_name_or_path, self.models["transformer"])
                        self.assertEqual(args.cache_latents, cache != "disabled")
                        self.assertEqual(args.mixed_precision, precision)
                        self.assertEqual(args.max_train_steps, 1500)

    def test_optional_flags_parse_for_both_modes(self):
        for mode in ("lora", "full"):
            config = self.config(training_type=mode, checkpoint_offload="unsloth", attention_mode="xformers",
                                 resume=str(self.root), save_state=True, sample_prompts="cat", sample_every_n_steps=10,
                                 t5_tokenizer_path=str(self.root), optimizer_settings={"optimizer_args": ["weight_decay=0.01"],
                                 "lr_warmup_steps": 10, "lr_scheduler": "constant_with_warmup"})
            run_dir = runner.prepare_run(config, self.root)
            args = self.parsers[mode].parse_args(json.loads((run_dir / "manifest.json").read_text())["argv"])
            self.assertTrue(args.unsloth_offload_checkpointing)
            self.assertTrue(args.save_state_on_train_end)
            self.assertEqual(args.optimizer_args, ["weight_decay=0.01"])


if __name__ == "__main__":
    unittest.main()
