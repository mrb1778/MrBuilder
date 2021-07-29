from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2


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


layer_attributes = {
    "kernelInitializer": kernel_initializer,
    "kernelRegularizer": kernel_regularizer
}


