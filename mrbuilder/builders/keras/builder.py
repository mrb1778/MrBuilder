from .builder_registry import KerasBuilderRegistry
import mrbuilder.builders.keras.layers.standard_layers
import mrbuilder.builders.keras.layers.custom_layers

from .layer_registry import registered_layers
# from .model_layers import model_layers
from .layers.layer_attributes import layer_attributes

builder = KerasBuilderRegistry(layers_builders=registered_layers)
# _builder_registry.add_layer_builders(standard_layers)
# _builder_registry.add_layer_builders(custom_layers)
# _builder_registry.add_layer_builders(model_layers)
builder.add_layer_attribute_builders(layer_attributes)