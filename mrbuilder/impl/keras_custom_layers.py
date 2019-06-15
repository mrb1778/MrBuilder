from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, Dropout, BatchNormalization, \
    Activation, LeakyReLU
from tensorflow.keras.regularizers import l2

import mrbuilder


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
        do_dropout=False,
        dropout_rate=0.4,
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

    if do_dropout:
        x = Dropout(dropout_rate)(x)

    return x


mrbuilder.register_layer_builder(
    "ConvActBN",
    lambda layer_options, layer_connection:
    conv_act_bn(x=layer_connection,
                size=layer_options("size"),
                kernel_size=layer_options("kernel", 3),
                strides=layer_options("strides", 1),
                dilation_rate=layer_options("dilation", (1, 1)),
                activation=layer_options("activation", "relu"),
                activation_alpha=layer_options("activationAlpha", 0.3),
                padding=layer_options("padding", "same"),
                momentum=layer_options("momentum", 0.99),
                do_dropout=layer_options("doDropout", False),
                dropout_rate=layer_options("dropoutRate", 0.4),
                do_batch_norm=layer_options("doBatchNorm", True),
                sep_conv=layer_options("sepConv", False),
                conv_transpose=layer_options("convTranspose", False),
                l2_weight_decay=layer_options("weightDecay")))