"""Execute only upstream argparse definitions, never imports or training code."""

import argparse
import ast
from pathlib import Path
import types


def definitions(path, names=None, **bindings):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and (node.name in names if names is not None else node.name.startswith("add_"))]
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *functions], type_ignores=[])
    namespace = {"argparse": argparse, "Path": Path, **bindings}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return types.SimpleNamespace(**{node.name: namespace[node.name] for node in functions})


def upstream_parsers(source):
    source = Path(source)
    library = source / "library"
    bindings = {name: definitions(library / filename) for name, filename in (
        ("args_util", "args.py"), ("config_util", "config_util.py"),
        ("deepspeed_utils", "deepspeed_utils.py"), ("sai_model_spec", "sai_model_spec.py"),
        ("custom_train_functions", "custom_train_functions.py"),
        ("anima_train_utils", "anima_train_utils.py"),
    )}
    bindings["add_logging_arguments"] = definitions(library / "utils.py", {"add_logging_arguments"}).add_logging_arguments
    bindings["add_custom_train_arguments"] = bindings["custom_train_functions"].add_custom_train_arguments
    bindings["train_network"] = definitions(source / "train_network.py", {"setup_parser"}, **bindings)
    return {mode: definitions(source / filename, {"setup_parser"}, **bindings).setup_parser()
            for mode, filename in (("lora", "anima_train_network.py"), ("full", "anima_train.py"))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    for mode, upstream_parser in upstream_parsers(args.source).items():
        print(f"{mode}: loaded {len(upstream_parser._actions)} argparse actions without importing sd-scripts")
