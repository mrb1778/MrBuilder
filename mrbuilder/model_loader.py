import json
from pathlib import Path
import os

import mrbuilder.builder_registry as br


def load_models_from_path(path) -> None:
    if os.path.isdir(path):
        for file_name in Path(path).glob("**/*.json"):
            load_models_from_file(file_name)
    else:
        load_models_from_file(path)


def load_models_from_file(file_path) -> None:
    with open(file_path) as file:
        parsed_path = json.load(file)

    if isinstance(parsed_path, list):
        for parsed_model in parsed_path:
            br.build_register_model(parsed_model)
    else:
        br.build_register_model(parsed_path)
