from typing import Callable

from mrbuilder.builder_registry import BuilderRegistry
from mrbuilder_pytorch.builder_models import PyTorchBuilderLayerInput, PyTorchBuilderModel


class PyTorchBuilderRegistry(BuilderRegistry):

    def get_model_creator(self) -> Callable:
        # def create_pytorch_model(inputs, layers, layers_by_name, output_config):
        # if output_config and "layers" in output_config:
        #     output_layer_names = output_config["layers"]
        #     if len(output_layer_names) == 1:
        #         outputs = layers_by_name[output_layer_names[0]]
        #     else:
        #         outputs = [layers_by_name[layer_name] for layer_name in layers_by_name]
        # else:
        #     outputs = layers[-1]
        # return PyTorchBuilderModel(inputs, layers, outputs)

        return PyTorchBuilderModel

    def get_model_input_builder(self) -> Callable:
        return lambda input_shape, _: PyTorchBuilderLayerInput(input_shape)


