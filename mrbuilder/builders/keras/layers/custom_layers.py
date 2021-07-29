from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, Dropout, BatchNormalization, \
    Activation, LeakyReLU
from tensorflow.keras.regularizers import l2

from ..layer_registry import register_layer

custom_layers = {}


def conv_act_bn(
        x,
        size,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="relu",
        activation_alpha=0.3,
        sep_conv=False,
        conv_transpose=False,
        kernel_regularizer=None,
        l2_weight_decay=None,
        momentum=0.99,
        dropout_rate=0.0,
        dilation_rate=(1, 1),
        do_batch_norm=True):
    if l2_weight_decay is not None:
        kernel_regularizer = l2(l2_weight_decay)

    if sep_conv:
        x = SeparableConv2D(
            size,
            kernel_size,
            padding=padding,
            strides=strides,
            kernel_regularizer=kernel_regularizer)(x)
    elif conv_transpose:
        x = Conv2DTranspose(
            size,
            kernel_size,
            padding=padding,
            strides=strides,
            kernel_regularizer=kernel_regularizer)(x)
    else:
        x = Conv2D(
            int(size),
            kernel_size,
            padding=padding,
            strides=strides,
            dilation_rate=dilation_rate,
            kernel_regularizer=kernel_regularizer)(x)

    if activation is not None:
        if activation == 'LeakyReLU':
            x = LeakyReLU(activation_alpha)(x)
        else:
            x = Activation(activation)(x)

    if do_batch_norm:
        x = BatchNormalization(momentum=momentum)(x)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    return x


@register_layer("ConvActBN")
def conv_act_bn_wrapper(config, connection):
    return conv_act_bn(x=connection,
                       size=config("size"),
                       kernel_size=config("kernel", 3),
                       strides=config("strides", 1),
                       dilation_rate=config("dilation", (1, 1)),
                       activation=config("activation", "relu"),
                       activation_alpha=config("activationAlpha",
                                               0.3),
                       padding=config("padding", "same"),
                       momentum=config("momentum", 0.99),
                       dropout_rate=config("dropoutRate", 0.0),
                       do_batch_norm=config("doBatchNorm", True),
                       sep_conv=config("sepConv", False),
                       conv_transpose=config("convTranspose", False),
                       l2_weight_decay=config("weightDecay"))
