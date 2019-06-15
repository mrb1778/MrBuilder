from typing import Callable, Dict

from mrbuilder.builder_config import BuilderConfig
from mrbuilder.expressions import sexpression
from mrbuilder.model_builder import ModelBuilder
from mrbuilder.utils.singleton import Singleton


class BuilderRegistry(metaclass=Singleton):
    builder_config: BuilderConfig

    layers_builders: Dict[str, Callable]
    layer_option_builders: Dict[str, Callable]

    expression_evaluator: Callable

    models: Dict[str, Callable]

    def __init__(self) -> None:
        super().__init__()
        self.layers_builders = {}
        self.layer_option_builders = {}

        self.expression_evaluator = sexpression.eval_expression

        self.builder_config = None
        self.models = {}

    def register_layer_builder(self, layer_type: str, layer_builder: Callable) -> None:
        self.layers_builders[layer_type] = layer_builder

    def get_layer_builder(self, layer_type: str) -> Callable:
        if layer_type in self.layers_builders:
            return self.layers_builders[layer_type]
        elif self.builder_config.has_layer_builder(layer_type):
            return self.builder_config.get_layer_builder(layer_type)
        else:
            raise MissingLayerTypeException("Unknown Layer Type: {} in layers: {}".format(layer_type, [*self.layers_builders.keys()]))

    def has_layer_builder(self, layer_type: str) -> bool:
        return layer_type in self.layers_builders or \
               self.builder_config.has_layer_builder(layer_type)

    def register_layer_options_builder(self, name: str, layer_option_builder: Callable) -> None:
        self.layer_option_builders[name] = layer_option_builder

    def get_layer_options_builder(self, name: str) -> Callable:
        if name in self.layer_option_builders:
            return self.layer_option_builders[name]
        elif self.builder_config.has_layer_options_builder(name):
            return self.builder_config.get_layer_options_builder(name)
        else:
            raise Exception("Unknown Layer Option: {}".format(name))

    def has_layer_options_builder(self, name: str) -> bool:
        return name in self.layer_option_builders or \
               self.builder_config.has_layer_options_builder(name)

    # Models
    def build_register_model(self, model_config: Dict, name: str = None) -> Callable:
        model_builder = ModelBuilder(self, self.builder_config, model_config).build()

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


builder_registry = BuilderRegistry()


def set_builder_config(builder_config: BuilderConfig, ignore_if_set: bool = False) -> None:
    if builder_registry.builder_config is None or not ignore_if_set:
        builder_registry.builder_config = builder_config


def get_model(name: str) -> Callable:
    return builder_registry.get_model(name)


def build_register_model(model_config: Dict, name: str = None) -> Callable:
    return builder_registry.build_register_model(model_config, name)


def register_layer_builder(layer_type: str, layer_builder: Callable) -> None:
    return builder_registry.register_layer_builder(layer_type, layer_builder)


class MissingModelException(Exception):
    pass


class MissingLayerTypeException(Exception):
    pass