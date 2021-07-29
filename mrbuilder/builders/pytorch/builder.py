from .builder_registry import PyTorchBuilderRegistry
from .layer_registry import registered_layers
import mrbuilder.builders.pytorch.layers.standard_layers
import mrbuilder.builders.pytorch.layers.custom_layers

builder = PyTorchBuilderRegistry(layers_builders=registered_layers)
