"""Exercise the actual node registrations with a minimal ComfyUI host interface."""

import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch, Mock

from test_anima import Fixture, ROOT


def load_nodes():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {key: None for key in ("diffusion_models", "unet", "text_encoders", "clip", "vae")}
    folder_paths.get_filename_list = lambda category: [f"{category}.safetensors"]
    name = "fluxtrainer_anima_test"
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT)]
    spec = importlib.util.spec_from_file_location(name + ".nodes_anima", ROOT / "nodes_anima.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: package, "folder_paths": folder_paths}):
        spec.loader.exec_module(module)
    return module, folder_paths


NODES, FOLDER_PATHS = load_nodes()


class NodeTests(Fixture):
    def test_registration_includes_legacy_and_anima_nodes(self):
        name = "fluxtrainer_registration_test"
        modules = {}
        for file in ("nodes", "nodes_sd3", "nodes_sdxl"):
            module = types.ModuleType(name + "." + file)
            module.NODE_CLASS_MAPPINGS = {file: object}
            module.NODE_DISPLAY_NAME_MAPPINGS = {file: file}
            modules[module.__name__] = module
        modules["folder_paths"] = FOLDER_PATHS
        spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        modules[name] = module
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        self.assertEqual(set(module.NODE_CLASS_MAPPINGS), {"nodes", "nodes_sd3", "nodes_sdxl", *NODES.NODE_CLASS_MAPPINGS})
        self.assertEqual(set(module.NODE_CLASS_MAPPINGS), set(module.NODE_DISPLAY_NAME_MAPPINGS))

    def test_all_node_defaults_build_valid_config(self):
        dataset = NODES.AnimaDatasetConfig().create_config(str(self.images), 1024, 1, 1, ".txt", "test character")[0]
        settings = {}
        for section in ("required", "optional"):
            for key, value in NODES.InitAnimaTraining.INPUT_TYPES()[section].items():
                if len(value) > 1 and "default" in value[1]:
                    settings[key] = value[1]["default"]
        result = NODES.InitAnimaTraining().configure(self.models, dataset, **settings)[0]
        self.assertEqual(result["training_type"], "lora")
        self.assertEqual(result["args"]["network_dim"], 16)

    def test_model_picker_uses_legacy_folder_aliases(self):
        with patch.object(FOLDER_PATHS, "folder_names_and_paths", {"unet": None, "clip": None, "vae": None}), \
             patch.object(FOLDER_PATHS, "get_full_path", create=True, side_effect=lambda category, name: self.models[name]):
            selected = NODES.AnimaModelSelect().select("transformer", "qwen3", "vae")[0]
        self.assertEqual(selected, self.models)

    def test_output_dir_strips_floyo_outputs_prefix(self):
        base = self.root / "comfy-output"
        cases = {
            "#outputs/Anima": base / "Anima",
            "output/Anima": base / "Anima",
            "(as-input)#outputs/Anima": base / "Anima",
            "#outputs": base,
            "output": base,
            "#outputs/anima": base / "anima",
            "anima": base / "anima",
            "/abs/Anima": Path("/abs/Anima"),
        }
        with patch.object(FOLDER_PATHS, "get_output_directory", create=True, return_value=str(base)):
            for raw, expected in cases.items():
                with self.subTest(raw=raw):
                    self.assertEqual(NODES.resolve_output_root(raw), expected)

    def test_training_reports_checkpoints_under_outputs(self):
        run_dir = self.root / "Anima" / "anima_lora-run"
        run_dir.mkdir(parents=True)
        (run_dir / "anima_lora.safetensors").write_bytes(b"lora")
        (run_dir / "anima_lora-step00001500.safetensors").write_bytes(b"step")
        result = {"model_path": str(run_dir / "anima_lora.safetensors"), "action": "train"}
        mm = types.ModuleType("comfy.model_management")
        mm.unload_all_models = Mock()
        mm.soft_empty_cache = Mock()
        mm.throw_exception_if_processing_interrupted = Mock()
        utils = types.ModuleType("comfy.utils")
        utils.ProgressBar = Mock()
        comfy = types.ModuleType("comfy")
        comfy.model_management, comfy.utils = mm, utils
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils}), \
             patch.object(FOLDER_PATHS, "get_output_directory", create=True, return_value=str(self.root)), \
             patch.object(NODES, "execute", return_value=(result, run_dir)) as execute:
            output = NODES.AnimaTrain().run(self.config(), action="train", output_dir="#outputs/Anima", unique_id="test-node-5")
        self.assertEqual(execute.call_args.args[1], self.root / "Anima")
        self.assertEqual(output["ui"]["files"], [
            {"filename": "anima_lora-step00001500.safetensors", "subfolder": "Anima/anima_lora-run", "type": "output"},
            {"filename": "anima_lora.safetensors", "subfolder": "Anima/anima_lora-run", "type": "output"},
        ])

    def test_validate_node_returns_ui_report_and_does_not_unload_models(self):
        mm = types.ModuleType("comfy.model_management")
        mm.unload_all_models = Mock(side_effect=AssertionError("validation must not unload models"))
        mm.soft_empty_cache = Mock()
        mm.throw_exception_if_processing_interrupted = Mock()
        utils = types.ModuleType("comfy.utils")
        utils.ProgressBar = Mock()
        comfy = types.ModuleType("comfy")
        comfy.model_management, comfy.utils = mm, utils
        result = {"model_path": "", "action": "validate"}
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils}), \
             patch.object(FOLDER_PATHS, "get_output_directory", create=True, return_value=str(self.root)), \
             patch.object(NODES, "execute", return_value=(result, self.root / "run")) as execute:
            output = NODES.AnimaTrain().run(self.config(), unique_id="test-node-4")
        self.assertEqual(output["result"][0], "")
        self.assertEqual(json.loads(output["ui"]["text"][0])["action"], "validate")
        self.assertEqual(execute.call_args.kwargs["action"], "validate")
        utils.ProgressBar.assert_called_once_with(1500, node_id="test-node-4")
        mm.unload_all_models.assert_not_called()

    def test_examples_only_contain_registered_nodes_and_default_to_validate(self):
        for path in (ROOT / "example_workflows").glob("anima*_api.json"):
            with self.subTest(path=path.name):
                graph = json.loads(path.read_text())
                for node in graph.values():
                    cls = NODES.NODE_CLASS_MAPPINGS[node["class_type"]]
                    required = cls.INPUT_TYPES()["required"]
                    self.assertTrue(set(required) <= node["inputs"].keys())
                    if node["class_type"] == "AnimaTrain":
                        self.assertEqual(node["inputs"]["action"], "validate")
                    for key, value in node["inputs"].items():
                        if isinstance(value, list):
                            source = graph[value[0]]
                            output = NODES.NODE_CLASS_MAPPINGS[source["class_type"]].RETURN_TYPES[value[1]]
                            self.assertEqual(required[key][0], output)

    def test_ui_examples_preserve_widgets_and_bidirectional_links(self):
        paths = list((ROOT / "example_workflows").glob("anima_*.json"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            if path.name.endswith("_api.json"):
                continue
            graph = json.loads(path.read_text())
            api = json.loads(path.with_name(path.stem + "_api.json").read_text())
            by_id = {node["id"]: node for node in graph["nodes"]}
            for link_id, source_id, source_slot, target_id, target_slot, socket_type in graph["links"]:
                self.assertIn(link_id, by_id[source_id]["outputs"][source_slot]["links"])
                self.assertEqual(by_id[target_id]["inputs"][target_slot]["link"], link_id)
                self.assertEqual(by_id[source_id]["outputs"][source_slot]["type"], socket_type)
            for node in graph["nodes"]:
                schema = NODES.NODE_CLASS_MAPPINGS[node["type"]].INPUT_TYPES()
                expected = []
                for section in ("required", "optional"):
                    for key in schema.get(section, {}):
                        value = api[str(node["id"])]["inputs"].get(key)
                        if value is not None and not isinstance(value, list):
                            expected.append(value)
                            if key == "seed":
                                expected.append("fixed")
                self.assertEqual(node["widgets_values"], expected)


if __name__ == "__main__":
    unittest.main()
