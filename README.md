# ![logo](./docs/images/logo.png) MrBuilder
Model and Repository Builder for Deep Learning with an emphasis on Keras / Tensorflow

```python
import mrbuilder_keras as mrb  # used to initialize Keras builder

mrb.load("/path/to/models")

model_builder = mrb.get_model("vgg16")


input_shape = [32, 32, 3]
model_params = {
    "initialDropoutRate": 0.35, 
    "dropoutRate": 0.45,
    "outputSize": 50
}
model = model_builder(input_shape, model_params)


model.summary()
```

vgg16.json
```json
{
  "name": "vgg16",
  "properties": {"initialConvSize":  64, "initialDropoutRate": 0.3, "dropoutRate": 0.4},
  "layers": [
    {"type": "ConvActBN", "size": "{{initialConvSize}}", "dropoutRate": "{{initialDropoutRate}}"},
    {"type": "ConvActBN", "size": "{{initialConvSize}}"},
    {"type": "MaxPooling2D", "size": 2},

    {"type": "ConvActBN", "size": "{{initialConvSize * 2}}", "dropoutRate": "{{dropoutRate}}"},
    {"type": "ConvActBN", "size": "{{initialConvSize * 2}}"},
    {"type": "MaxPooling2D", "size": 2},

    {"type": "ConvActBN", "size": "{{initialConvSize * 4}}", "dropoutRate": "{{dropoutRate}}"},
    {"type": "ConvActBN", "size": "{{initialConvSize * 4}}", "dropoutRate": "{{dropoutRate}}"},
    {"type": "ConvActBN", "size": "{{initialConvSize * 4}}"},
    {"type": "MaxPooling2D", "size": 2},

    {"type": "ConvActBN", "size": "{{initialConvSize * 8}}", "dropoutRate": "{{dropoutRate}}"},
    {"type": "ConvActBN", "size": "{{initialConvSize * 8}}", "dropoutRate": "{{dropoutRate}}"},
    {"type": "ConvActBN", "size": "{{initialConvSize * 8}}", "dropoutRate": "{{dropoutRate}}"},
    {"type": "ConvActBN", "size": "{{initialConvSize * 8}}"},
    {"type": "MaxPooling2D", "size": 2},
    {"type": "ConvActBN", "size": "{{initialConvSize * 8}}", "dropoutRate": "{{dropoutRate}}"},
    {"type": "ConvActBN", "size": "{{initialConvSize * 8}}", "dropoutRate": "{{dropoutRate}}"},
    {"type": "ConvActBN", "size": "{{initialConvSize * 8}}"},
    {"type": "MaxPooling2D", "size": 2},
    {"type": "Dropout", "rate": "{{dropoutRate}}"},

    {"type": "Flatten"},
    {"type": "Dense", "size": "{{initialConvSize * 8}}"},
    {"type": "Activation", "function": "softmax"},
    {"type": "BatchNormalization"},

    {"type": "Dropout", "rate": "{{dropoutRate}}"},
    {"type": "Dense", "size": "{{outputSize}}"},
    {"type": "Activation", "function": "softmax"}
  ]
}
```

### TODO
* Core
  * Looping
  * Global Layer Group Definition
  * Template multi layer default params  
  * Model as first class layer
  * Decorator Layer Registration
* Documentation
  * API HTML Documentation
  * Inline Documentation
  * Model Definition Documentation
  * Sample model usages
* Global
  * Web Based WYSIWIG Model Builder
  * Expand expression syntax

* PyTorch implementation
* JavaScript implementation
* Java implementation

