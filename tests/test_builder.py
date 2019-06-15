import unittest

import mrbuilder
import mrbuilder.impl.keras_builder
from mrbuilder.builder_registry import MissingLayerTypeException


class BuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.model_definition = {
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
        mrbuilder.builder_registry.build_register_model(cls.model_definition)
        cls.input_shape = [32, 32, 3]

        cls.base_params = {
            "outputSize": 50,
            "optimizer": "adam",
            "learning_rate": 2e-4,
            "beta_1": 0.5,
            "loss_fn": "mse"
        }

    def test_model_build_basic(self):
        model_builder = mrbuilder.get_model("vgg16")
        model = model_builder(self.input_shape, self.base_params)
        layers = model.layers
        self.assertEqual(len(layers),
                         65,
                         "number of layers is not correct")
        # noinspection PyTypeChecker
        self.assertEqual(layers[0].input.shape.as_list(),
                         [None] + self.input_shape,
                         "input shape is not correct")
        self.assertEqual(layers[-1].output.shape[1],
                         self.base_params["outputSize"],
                         "output shape is not correct")

    def test_model_build_expression(self):
        model_definition = {
            "name": "testModelExpression",
            "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
            "layers": [
                {"type": "Conv2D", "size": "{{64 * 2}}"},
                {"type": "Dropout", "rate": "initialDropoutRate"},
                {"type": "ConvActBN", "size": 64},
                {"type": "MaxPooling2D", "size": 2},

                {"type": "Dropout", "rate": "dropoutRate"},
                {"type": "Dense", "size": "outputSize"},
                {"type": "Activation", "function": "softmax"}
            ]
        }
        mrbuilder.build_register_model(model_definition)
        model_builder = mrbuilder.get_model(model_definition["name"])
        conv_size = 64
        model = model_builder(self.input_shape, {
            **self.base_params,
            'convSize': conv_size
        })
        layers = model.layers
        self.assertEqual(layers[1].output.shape[-1],
                         conv_size * 2,
                         "output shape is not correct")

    def test_model_layer_missing(self):
        model_definition_fail = {
            "name": "testModelFail",
            "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
            "layers": [
                {"type": "MISSING_LAYER_TYPE", "size": 64},
                {"type": "Dropout", "rate": "initialDropoutRate"},
                {"type": "ConvActBN", "size": 64},
                {"type": "MaxPooling2D", "size": 2},

                {"type": "Dropout", "rate": "dropoutRate"},
                {"type": "Dense", "size": "outputSize"},
                {"type": "Activation", "function": "softmax"}
            ]
        }
        with self.assertRaises(MissingLayerTypeException, msg="Layer type was missing, but not caught"):
            mrbuilder.build_register_model(model_definition_fail)
            model_builder = mrbuilder.get_model(model_definition_fail["name"])
            model_builder(self.input_shape, self.base_params)


if __name__ == '__main__':
    unittest.main()
