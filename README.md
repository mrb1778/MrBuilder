# MrBuilder
Builder and Bootstrap Library for Deep Learning with an emphasis on Keras / Tensorflow

```
from mrbuilder import load_models_from_path, get_model
from mrbuilder.keras_builders import *

load_models_from_path(args.models)

vgg_builder = get_model("vgg16")
vgg = vgg_builder(input_shape, {"initialDropoutRate": 0.3, "dropoutRate": 0.4}, outputSize)
vgg.summary()

```

vgg16.json
```
{
  "name": "vgg16",
  "layers": [
    {"type": "conv_act_bn", "size": 64},
    {"type": "Dropout", "rate": "initialDropoutRate"},
    {"type": "conv_act_bn", "size": 64},
    {"type": "MaxPooling2D", "size": 2},

    {"type": "conv_act_bn", "size": 128},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "conv_act_bn", "size": 128},
    {"type": "MaxPooling2D", "size": 2},

    {"type": "conv_act_bn", "size": 256},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "conv_act_bn", "size": 256},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "conv_act_bn", "size": 256},
    {"type": "MaxPooling2D", "size": 2},

    {"type": "conv_act_bn", "size": 512},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "conv_act_bn", "size": 512},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "conv_act_bn", "size": 512},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "conv_act_bn", "size": 512},
    {"type": "MaxPooling2D", "size": 2},
    {"type": "conv_act_bn", "size": 512},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "conv_act_bn", "size": 512},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "conv_act_bn", "size": 512},
    {"type": "MaxPooling2D", "size": 2},
    {"type": "Dropout", "rate": "dropoutRate"},

    {"type": "Flatten"},
    {"type": "Dense", "size": 512},
    {"type": "Activation", "function": "softmax"},
    {"type": "BatchNormalization"},

    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "Dense", "size": "outputSize"},
    {"type": "Activation", "function": "softmax"}
  ]
}

```
