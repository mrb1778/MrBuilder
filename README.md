# MrBuilder
Model and Repository Builder for Deep Learning with an emphasis on Keras / Tensorflow

```python
import mrbuilder
import mrbuilder.impl.keras_builder  # used to initialize Keras baackend

mrbuilder.load_models_from_path("/path/to/models")

model_builder = mrbuilder.get_model("vgg16")


input_shape = [32, 32, 3]
model_params = {
    "outputSize": 50,
    "optimizer": "adam",
    "learning_rate": 2e-4,
    "beta_1": 0.5,
    "loss_fn": "mse"
}
model = model_builder(input_shape, model_params)


model.summary()
```

vgg16.json
```json
{
  "name": "vgg16",
  "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
  "layers": [
    {"type": "ConvActBN", "size": 64},
    {"type": "Dropout", "rate": "initialDropoutRate"},
    {"type": "ConvActBN", "size": 64},
    {"type": "MaxPooling2D", "size": 2},

    {"type": "ConvActBN", "size": 128},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "ConvActBN", "size": 128},
    {"type": "MaxPooling2D", "size": 2},

    {"type": "ConvActBN", "size": 256},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "ConvActBN", "size": 256},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "ConvActBN", "size": 256},
    {"type": "MaxPooling2D", "size": 2},

    {"type": "ConvActBN", "size": 512},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "ConvActBN", "size": 512},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "ConvActBN", "size": 512},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "ConvActBN", "size": 512},
    {"type": "MaxPooling2D", "size": 2},
    {"type": "ConvActBN", "size": 512},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "ConvActBN", "size": 512},
    {"type": "Dropout", "rate": "dropoutRate"},
    {"type": "ConvActBN", "size": 512},
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
