from typing import Dict, Callable

from mrbuilder_pytorch.builder_registry import PyTorchBuilderRegistry
from mrbuilder_pytorch.layer_registry import registered_layers
import mrbuilder_pytorch.layers.standard_layers
import mrbuilder_pytorch.layers.custom_layers

_builder_registry = PyTorchBuilderRegistry(layers_builders=registered_layers)


def register_layer_builder(layer_type: str, layer_builder: Callable) -> None:
    _builder_registry.add_layer_builder(layer_type, layer_builder)


def register_layer_builders(layers: Dict[str, Callable]) -> None:
    _builder_registry.add_layer_builders(layers)


def build(model_config: Dict, name: str = None) -> Callable:
    return _builder_registry.build(model_config, name)


def load(path: str = None) -> None:
    _builder_registry.load(path)


def get_model(name: str) -> Callable:
    """Deprecated use get or get_model_builder"""
    return get_model_builder(name)


def get(name: str) -> Callable:
    return get_model_builder(name)


def get_model_builder(name: str) -> Callable:
    return _builder_registry.get_model_builder(name)
