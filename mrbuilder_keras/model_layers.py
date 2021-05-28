from tensorflow.keras.applications import Xception


model_layers = {}


def create_xception_layer(x, weights='imagenet', include_top=False, classes=100, activation='softmax'):
    model = Xception(
        input_tensor=x,
        include_top=include_top,
        classes=classes,
        classifier_activation=activation,
        weights=weights
    )
    return model.output


model_layers["Xception"] = lambda config, connection: create_xception_layer(x=connection,
                                                                    size=config("size"),
                                                                    weights=config("weights", "imagenet"),
                                                                    include_top=config("kernel", 3),
                                                                    strides=config("strides", 1),
                                                                    dilation_rate=config("dilation", (1, 1)),
                                                                    activation_alpha=config("activationAlpha",
                                                                                            0.3),
                                                                    padding=config("padding", "same"),
                                                                    momentum=config("momentum", 0.99),
                                                                    dropout_rate=config("dropoutRate", 0.0),
                                                                    do_batch_norm=config("doBatchNorm", True),
                                                                    sep_conv=config("sepConv", False),
                                                                    conv_transpose=config("convTranspose", False),
                                                                    l2_weight_decay=config("weightDecay"))