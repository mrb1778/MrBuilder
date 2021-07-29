from collections import Callable
from typing import Dict

from .builder import builder

# Public Interface Methods
def register_layer_builder(layer_type: str, layer_builder: Callable) -> None:
    builder.add_layer_builder(layer_type, layer_builder)


def register_layer_builders(layers: Dict[str, Callable]) -> None:
    builder.add_layer_builders(layers)


def build(model_config: Dict, name: str = None) -> Callable:
    return builder.build(model_config, name)


def load(path: str = None) -> None:
    builder.load(path)


def get_model(name: str) -> Callable:
    """Deprecated use get or get_model_builder"""
    return get_model_builder(name)


def get(name: str) -> Callable:
    return get_model_builder(name)


def get_model_builder(name: str) -> Callable:
    return builder.get_model_builder(name)