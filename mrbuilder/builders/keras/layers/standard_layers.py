from ..layer_registry import layer

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, Flatten, Dense, Activation, \
    LeakyReLU, BatchNormalization, GlobalAveragePooling2D, UpSampling2D, Reshape, Concatenate, Add, LSTM


@layer("Conv2D")
def mrb_layer_Conv2D(config, connection):
    return Conv2D(
        filters=config("size"),
        kernel_size=config("kernel", 3, repeat=2),
        strides=config("strides", 1, repeat=2),
        padding=config("padding", "same"),
        dilation_rate=config("dilation", (1, 1)),
        activation=config("activation"),
        kernel_initializer=config("kernelInitializer", "glorot_uniform")
    )(connection)


@layer("Conv2DTranspose")
def mrb_layer_Conv2DTranspose(config, connection):
    return Conv2DTranspose(
        filters=config("size"),
        kernel_size=config("kernel", 3),
        strides=config("strides", 2),
        padding=config("padding", "same"),
        activation=config("activation"),
        kernel_initializer=config("kernelInitializer", "glorot_uniform")
    )(connection)


@layer("Dropout")
def mrb_layer_Dropout(config, connection):
    return Dropout(config("rate", 0.25))(connection)


@layer("MaxPooling2D")
def mrb_layer_MaxPooling2D(config, connection):
    return MaxPooling2D(
        pool_size=config("size", 2, repeat=2),
        strides=config("strides", 1, repeat=2)
    )(connection)


@layer("Flatten")
def mrb_layer_Flatten(config, connection):
    return Flatten()(connection)


@layer("Dense")
def mrb_layer_Dense(config, connection):
    return Dense(
        config("size"),
        activation=config("activation"),
        kernel_regularizer=config("kernelRegularizer"),
        kernel_initializer=config("kernelInitializer", "glorot_uniform")
    )(connection)


@layer("Activation")
def mrb_layer_Activation(config, connection):
    return Activation(config("function"))(connection)


@layer("LeakyReLU")
def mrb_layer_LeakyReLU(config, connection):
    return LeakyReLU(alpha=config("alpha", 0.3))(connection)


@layer("BatchNormalization", "BatchNorm2d", "BatchNorm")
def mrb_layer_BatchNormalization(config, connection):
    return BatchNormalization(momentum=config("momentum", 0.99))(connection)


@layer("GlobalAveragePooling2D")
def mrb_layer_GlobalAveragePooling2D(config, connection):
    return GlobalAveragePooling2D()(connection)


@layer("UpSampling2D")
def mrb_layer_UpSampling2D(config, connection):
    return UpSampling2D(config("size", 2))(connection)


@layer("Reshape")
def mrb_layer_Reshape(config, connection):
    return Reshape(target_shape=config("shape"))(connection)


@layer("Concatenate")
def mrb_layer_Concatenate(config, connection):
    return Concatenate(axis=config("axis", -1))(connection)


@layer("Add")
def mrb_layer_Add(config, connection):
    return Add()(connection)


@layer("LSTM")
def mrb_layer_LSTM(config, connection):
    return LSTM(
        units=config("size"),
        return_sequences=config("more", False)
    )(connection)
