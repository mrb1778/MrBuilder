from typing import Callable

from tensorflow.keras import Input, Model

from mrbuilder.builder_registry import BuilderRegistry


class KerasBuilderRegistry(BuilderRegistry):
    def get_model_creator(self) -> Callable:
        def create_keras_model(inputs, layers, layers_by_name, output_config, name=None):
            if output_config and "layers" in output_config:
                output_layer_names = output_config["layers"]
                if len(output_layer_names) == 1:
                    outputs = layers_by_name[output_layer_names[0]]
                else:
                    outputs = [layers_by_name[layer_name] for layer_name in layers_by_name]
            else:
                outputs = layers[-1]

            return Model(inputs=inputs, outputs=outputs, name=name)

        return create_keras_model

    def get_model_input_builder(self) -> Callable:
        return lambda input_shape, _: Input(shape=input_shape)