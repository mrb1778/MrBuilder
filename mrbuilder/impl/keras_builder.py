from typing import Callable, Dict

from tensorflow.keras import Model, Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, Flatten, Dense, Activation, \
    LeakyReLU, BatchNormalization, GlobalAveragePooling2D, UpSampling2D, Reshape, Concatenate, Add
from tensorflow.keras.regularizers import l2

from mrbuilder.utils.singleton import Singleton
from mrbuilder.builder_config import BuilderConfig
from mrbuilder.builder_registry import set_builder_config
import mrbuilder.impl.keras_custom_layers


class KerasBuilderConfig(BuilderConfig, metaclass=Singleton):
    def __init__(self) -> None:
        super().__init__()
        self.layers_builders.update(self.get_layer_builders())
        self.layer_option_builders.update(self.get_layer_options_builders())

    def get_model_creator(self) -> Callable:
        def create_keras_model(inputs, layers, layers_by_name, output_config):
            if output_config and "layers" in output_config:
                output_layer_names = output_config["layers"]
                if len(output_layer_names) == 1:
                    outputs = layers_by_name[output_layer_names[0]]
                else:
                    outputs = [layers_by_name[layer_name] for layer_name in layers_by_name]
            else:
                outputs = layers[-1]

            return Model(inputs=inputs, outputs=outputs)

        return create_keras_model

    def get_model_initializer(self) -> Callable:
        return lambda input_shape, _: Input(shape=input_shape)

    def get_layer_builders(self) -> Dict[str, Callable]:
        return {
            "Conv2D": lambda layer_options, layer_connection:
                Conv2D(
                    layer_options("size"),
                    kernel_size=layer_options("kernel", 3),
                    strides=layer_options("strides", 1),
                    activation=layer_options("activation"),
                    padding=layer_options("padding", "same"),
                    dilation_rate=layer_options("dilation", (1, 1)),
                    kernel_initializer=layer_options("kernelInitializer", "glorot_uniform")
                )(layer_connection),
            "Conv2DTranspose": lambda layer_options, layer_connection:
                Conv2DTranspose(
                    layer_options("size"),
                    kernel_size=layer_options("kernel", 3),
                    strides=layer_options("strides", 2),
                    padding=layer_options("padding", "same"),
                    activation=layer_options("activation"),
                    kernel_initializer=layer_options("kernelInitializer", "glorot_uniform")
                )(layer_connection),
            "Dropout": lambda layer_options, layer_connection:
                Dropout(layer_options("rate", 0.25))(layer_connection),
            "MaxPooling2D": lambda layer_options, layer_connection:
                MaxPooling2D(
                    pool_size=layer_options("size", 2, repeat=2),
                    strides=layer_options("strides", 1, repeat=2)
                )(layer_connection),
            "Flatten": lambda layer_options, layer_connection:
                Flatten()(layer_connection),
            "Dense": lambda layer_options, layer_connection:
                Dense(
                    layer_options("size"),
                    activation=layer_options("activation"),
                    kernel_regularizer=layer_options("kernelRegularizer"),
                    kernel_initializer=layer_options("kernelInitializer", "glorot_uniform")
                )(layer_connection),
            "Activation": lambda layer_options, layer_connection:
                Activation(layer_options("function"))(layer_connection),
            "LeakyReLU": lambda layer_options, layer_connection:
                LeakyReLU(alpha=layer_options("alpha", 0.3))(layer_connection),
            "BatchNormalization": lambda layer_options, layer_connection:
                BatchNormalization(momentum=layer_options("momentum", 0.99))(layer_connection),
            "GlobalAveragePooling2D": lambda layer_options, layer_connection:
                GlobalAveragePooling2D()(layer_connection),
            "UpSampling2D": lambda layer_options, layer_connection:
                UpSampling2D(layer_options("size", 2))(layer_connection),
            "Reshape": lambda layer_options, layer_connection:
                Reshape(target_shape=layer_options("shape"))(layer_connection),
            "Concatenate": lambda layer_options, layer_connection:
                Concatenate()(layer_connection),
            "Add": lambda layer_options, layer_connection:
                Add()(layer_connection)
        }

    def get_layer_options_builders(self) -> Dict[str, Callable]:
        def kernel_initializer(layer_options, default_value):
            option_value = layer_options("kernelInitializer")
            if option_value == "RandomNormal":
                return RandomNormal(stddev=layer_options("kernelInitializerStDev", 0.05))
            else:
                return default_value

        def kernel_regularizer(layer_options, default_value):
            option_value = layer_options("kernelRegularizer")
            if option_value == "l2":
                return l2(layer_options("l2WeightDecay", 0.01))
            else:
                return default_value

        return {
            "kernelInitializer": kernel_initializer,
            "kernelRegularizer": kernel_regularizer
        }


set_builder_config(KerasBuilderConfig(), True)

