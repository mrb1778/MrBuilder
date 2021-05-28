from typing import Dict, Callable

from mrbuilder_keras.builder_registry import KerasBuilderRegistry
import mrbuilder_keras.standard_layers
import mrbuilder_keras.custom_layers
from mrbuilder_keras.layer_registry import registered_layers
from mrbuilder_keras.model_layers import model_layers
from mrbuilder_keras.layer_attributes import layer_attributes

_builder_registry = KerasBuilderRegistry(layers_builders=registered_layers)
# _builder_registry.add_layer_builders(standard_layers)
# _builder_registry.add_layer_builders(custom_layers)
# _builder_registry.add_layer_builders(model_layers)
_builder_registry.add_layer_attribute_builders(layer_attributes)


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

