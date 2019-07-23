from typing import Callable, Dict

from tensorflow.keras import Model, Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, Flatten, Dense, Activation, \
    LeakyReLU, BatchNormalization, GlobalAveragePooling2D, UpSampling2D, Reshape, Concatenate, Add, LSTM
from tensorflow.keras.regularizers import l2

from mrbuilder.utils.singleton import Singleton
from mrbuilder.builder_config import BuilderConfig
from mrbuilder.builder_registry import set_builder_config
import mrbuilder.impl.keras_custom_layers


class KerasBuilderConfig(BuilderConfig, metaclass=Singleton):
    def __init__(self) -> None:
        super().__init__()
        self.layers_builders.update(self.get_layer_builders())
        self.layer_attribute_builders.update(self.get_layer_attribute_builders())

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

    def get_model_input_builder(self) -> Callable:
        return lambda input_shape, _: Input(shape=input_shape)

    def get_layer_builders(self) -> Dict[str, Callable]:
        return {
            "Conv2D": lambda config, connection:
                Conv2D(
                    filters=config("size"),
                    kernel_size=config("kernel", 3, repeat=2),
                    strides=config("strides", 1, repeat=2),
                    padding=config("padding", "same"),
                    dilation_rate=config("dilation", (1, 1)),
                    activation=config("activation"),
                    kernel_initializer=config("kernelInitializer", "glorot_uniform")
                )(connection),
            "Conv2DTranspose": lambda config, connection:
                Conv2DTranspose(
                    filters=config("size"),
                    kernel_size=config("kernel", 3),
                    strides=config("strides", 2),
                    padding=config("padding", "same"),
                    activation=config("activation"),
                    kernel_initializer=config("kernelInitializer", "glorot_uniform")
                )(connection),
            "Dropout": lambda config, connection:
                Dropout(config("rate", 0.25))(connection),
            "MaxPooling2D": lambda config, connection:
                MaxPooling2D(
                    pool_size=config("size", 2, repeat=2),
                    strides=config("strides", 1, repeat=2)
                )(connection),
            "Flatten": lambda config, connection:
                Flatten()(connection),
            "Dense": lambda config, connection:
                Dense(
                    config("size"),
                    activation=config("activation"),
                    kernel_regularizer=config("kernelRegularizer"),
                    kernel_initializer=config("kernelInitializer", "glorot_uniform")
                )(connection),
            "Activation": lambda config, connection:
                Activation(config("function"))(connection),
            "LeakyReLU": lambda config, connection:
                LeakyReLU(alpha=config("alpha", 0.3))(connection),
            "BatchNormalization": lambda config, connection:
                BatchNormalization(momentum=config("momentum", 0.99))(connection),
            "GlobalAveragePooling2D": lambda config, connection:
                GlobalAveragePooling2D()(connection),
            "UpSampling2D": lambda config, connection:
                UpSampling2D(config("size", 2))(connection),
            "Reshape": lambda config, connection:
                Reshape(target_shape=config("shape"))(connection),
            "Concatenate": lambda config, connection:
                Concatenate(axis=config("axis", -1))(connection),
            "Add": lambda config, connection:
                Add()(connection),
            "LSTM": lambda config, connection:
                LSTM(
                    units=config("size"),
                    return_sequences=config("returnSequences", False)
                )(connection)
        }

    def get_layer_attribute_builders(self) -> Dict[str, Callable]:
        def kernel_initializer(config, default_value):
            option_value = config("kernelInitializer")
            if option_value == "RandomNormal":
                return RandomNormal(stddev=config("kernelInitializerStDev", 0.05))
            else:
                return default_value

        def kernel_regularizer(config, default_value):
            option_value = config("kernelRegularizer")
            if option_value == "l2":
                return l2(config("l2WeightDecay", 0.01))
            else:
                return default_value

        return {
            "kernelInitializer": kernel_initializer,
            "kernelRegularizer": kernel_regularizer
        }


set_builder_config(KerasBuilderConfig(), True)

