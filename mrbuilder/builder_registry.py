import json
import os
from pathlib import Path
from typing import Callable, Dict

from mrbuilder.expressions.sexpression import SimpleExpressionEvaluator
from mrbuilder.model_builder import ModelBuilder, MissingLayerTypeException


class BuilderRegistry:
    layers_builders: Dict[str, Callable]
    layer_attribute_builders: Dict[str, Callable]

    expression_evaluator: SimpleExpressionEvaluator

    model_builders: Dict[str, Callable]

    def __init__(self, layers_builders=None, layer_attribute_builders=None) -> None:
        super().__init__()
        self.layers_builders = layers_builders if layers_builders is not None else {}
        self.layer_attribute_builders = layer_attribute_builders if layer_attribute_builders is not None else {}

        self.expression_evaluator = SimpleExpressionEvaluator()

        self.model_builders = {}

    def get_model_creator(self) -> Callable:
        pass

    def get_model_input_builder(self) -> Callable:
        pass

    def add_layer_builder(self, layer_type: str, layer_builder: Callable) -> None:
        self.layers_builders[layer_type] = layer_builder

    def add_layer_builders(self, layer_builders: Dict[str, Callable]) -> None:
        for layer_type, layer_builder in layer_builders.items():
            self.add_layer_builder(layer_type, layer_builder)

    def get_layer_builder(self, layer_type: str) -> Callable:
        layer_type = layer_type.lower()
        if layer_type in self.layers_builders:
            return self.layers_builders[layer_type]
        else:
            raise MissingLayerTypeException(
                f"Unknown Layer Type: {layer_type}.  Valid Types are {self.layers_builders.keys()}")

    def has_layer_builder(self, layer_type: str) -> bool:
        return layer_type in self.layers_builders

    def add_layer_attribute_builder(self, name: str, layer_attribute_builder: Callable) -> None:
        self.layer_attribute_builders[name] = layer_attribute_builder

    def add_layer_attribute_builders(self, attribute_builders: Dict[str, Callable]) -> None:
        for attribute_type, attribute_builder in attribute_builders.items():
            self.add_layer_attribute_builder(attribute_type, attribute_builder)

    def get_layer_attribute_builder(self, name: str) -> Callable:
        if name in self.layer_attribute_builders:
            return self.layer_attribute_builders[name]
        else:
            raise Exception(f"Unknown Layer Option: {name}")

    def has_layer_attribute_builder(self, name: str) -> bool:
        return name in self.layer_attribute_builders

    # Models
    def build(self, model_config: Dict, name: str = None, register: bool = True) -> Callable:
        model_name = name if name is not None else model_config["name"]
        model_builder = ModelBuilder(self, model_config, name=model_name).build()
        if register:
            self.register_model_builder(model_name, model_builder)

        return model_builder

    def register_model_builder(self, name: str, model_builder: Callable) -> None:
        self.model_builders[name] = model_builder

    def get_model_builder(self, name: str) -> Callable:
        if name in self.model_builders:
            return self.model_builders[name]
        else:
            raise MissingModelBuilderException(
                f"Model Builder not found {name} in model builders: {[*self.model_builders.keys()]}")

    def is_model_builder_registered(self, name: str) -> bool:
        return name in self.model_builders

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
            for parsed_model_builder in parsed_path:
                self.build(parsed_model_builder)
        else:
            self.build(parsed_path)


class MissingModelBuilderException(Exception):
    pass
