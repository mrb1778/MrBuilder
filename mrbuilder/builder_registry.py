import json
import os
from pathlib import Path
from typing import Callable, Dict

from mrbuilder.builder_config import BuilderConfig
from mrbuilder.expressions.sexpression import SimpleExpressionEvaluator
from mrbuilder.model_builder import ModelBuilder, MissingLayerTypeException


class BuilderRegistry:
    builder_config: BuilderConfig

    layers_builders: Dict[str, Callable]
    layer_attribute_builders: Dict[str, Callable]

    expression_evaluator: SimpleExpressionEvaluator

    models: Dict[str, Callable]

    def __init__(self, builder_config: BuilderConfig) -> None:
        super().__init__()
        self.layers_builders = {}
        self.layer_attribute_builders = {}

        self.expression_evaluator = SimpleExpressionEvaluator()

        self.builder_config = builder_config
        self.models = {}

    def register_layer_builder(self, layer_type: str, layer_builder: Callable) -> None:
        self.layers_builders[layer_type] = layer_builder

    def register_layer_builders(self, layers: Dict[str, Callable]) -> None:
        for layer_type, layer_builder in layers.items():
            self.register_layer_builder(layer_type, layer_builder)

    def get_layer_builder(self, layer_type: str) -> Callable:
        if layer_type in self.layers_builders:
            return self.layers_builders[layer_type]
        elif self.builder_config.has_layer_builder(layer_type):
            return self.builder_config.get_layer_builder(layer_type)
        else:
            raise MissingLayerTypeException(
                "Unknown Layer Type: {} in layers: {}".format(layer_type, [*self.layers_builders.keys()]))

    def has_layer_builder(self, layer_type: str) -> bool:
        return layer_type in self.layers_builders or \
               self.builder_config.has_layer_builder(layer_type)

    def register_layer_attribute_builder(self, name: str, layer_attribute_builder: Callable) -> None:
        self.layer_attribute_builders[name] = layer_attribute_builder

    def get_layer_attribute_builder(self, name: str) -> Callable:
        if name in self.layer_attribute_builders:
            return self.layer_attribute_builders[name]
        elif self.builder_config.has_layer_attribute_builder(name):
            return self.builder_config.get_layer_attribute_builder(name)
        else:
            raise Exception("Unknown Layer Option: {}".format(name))

    def has_layer_attribute_builder(self, name: str) -> bool:
        return name in self.layer_attribute_builders or \
               self.builder_config.has_layer_attribute_builder(name)

    # Models
    def build(self, model_config: Dict, name: str = None, register: bool = True) -> Callable:
        model_builder = ModelBuilder(self, self.builder_config, model_config).build()
        if register:
            model_name = name if name is not None else model_config["name"]
            self.register_model(model_name, model_builder)

        return model_builder

    def register_model(self, name: str, model_builder: Callable) -> None:
        self.models[name] = model_builder

    def get_model(self, name: str) -> Callable:
        if name in self.models:
            return self.models[name]
        else:
            raise MissingModelException("Model not found {} in models: {}".format(name, [*self.models.keys()]))

    def is_model_registered(self, name: str) -> bool:
        return name in self.models

    # loading  (moved from model_loader)
    def load(self, path: str = None) -> None:
        if path is None:
            path = str(Path(__file__).parent.parent.joinpath('models'))

        if os.path.isdir(path):
            for file_name in Path(path).glob("**/*.json"):
                self.load_file(file_name)
        else:
            self.load_file(path)

    def load_file(self, file_path) -> None:
        with open(file_path) as file:
            parsed_path = json.load(file)

        if isinstance(parsed_path, list):
            for parsed_model in parsed_path:
                self.build(parsed_model)
        else:
            self.build(parsed_path)


class MissingModelException(Exception):
    pass
