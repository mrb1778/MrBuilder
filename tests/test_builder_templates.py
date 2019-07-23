import unittest

import mrbuilder
import mrbuilder.impl.keras_builder
from mrbuilder.model_builder import MissingLayerTypeException
from mrbuilder.variable_registry import VariableNotFoundException


class BuilderTemplatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.input_shape = [32, 32, 3]

        cls.base_params = {
            "outputSize": 50
        }

    def test_model_build_template_simple(self):
        model_definition = {
            "name": "testModelTemplateSimple",
            "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
            "templates": [
                {"name": "template1", "type": "Conv2D", "strides": 2}
            ],
            "layers": [
                {"template": "template1", "size": 64},
                {"type": "Conv2D", "size": 64, "strides": 2}
            ]
        }
        mrbuilder.build_register_model(model_definition)
        model_builder = mrbuilder.get_model(model_definition["name"])
        model = model_builder(self.input_shape)
        layers = model.layers
        self.assertEqual(layers[1].strides[0],
                         model_definition["templates"][0]["strides"],
                         "template field is not correct")

        self.assertEqual(layers[1].__class__.__name__,
                         layers[2].__class__.__name__,
                         "wrong template type")

    def test_model_template_missing(self):
        model_definition_fail = {
            "name": "testModelFail",
            "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
            "layers": [
                {"type": "MISSING_LAYER_TYPE", "size": 64},
                {"type": "Dropout", "rate": "{{initialDropoutRate}}"},
                {"type": "ConvActBN", "size": 64},
                {"type": "MaxPooling2D", "size": 2},

                {"type": "Dropout", "rate": "{{dropoutRate}}"},
                {"type": "Dense", "size": "{{outputSize}}"},
                {"type": "Activation", "function": "softmax"}
            ]
        }
        with self.assertRaises(MissingLayerTypeException, msg="Layer type was missing, but not caught"):
            mrbuilder.build_register_model(model_definition_fail)
            model_builder = mrbuilder.get_model(model_definition_fail["name"])
            model_builder(self.input_shape, self.base_params)

    def test_template_var_missing(self):
        model_definition_fail = {
            "name": "testModelFail",
            "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
            "layers": [
                {"type": "Conv2D", "size": "{{missingProperty}}"},
                {"type": "Dropout", "rate": "{{initialDropoutRate}}"},
                {"type": "ConvActBN", "size": 64},
                {"type": "MaxPooling2D", "size": 2},

                {"type": "Dropout", "rate": "{{dropoutRate}}"},
                {"type": "Dense", "size": "{{outputSize}}"},
                {"type": "Activation", "function": "softmax"}
            ]
        }
        with self.assertRaises(VariableNotFoundException, msg="Variable was missing, but not caught"):
            mrbuilder.build_register_model(model_definition_fail)
            model_builder = mrbuilder.get_model(model_definition_fail["name"])
            model_builder(self.input_shape, self.base_params)

    def test_model_build_template_multi_layers_simple(self):
        model_definition = {
            "name": "testModelTemplateSimple",
            "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
            "templates": [
                {"name": "template1", "layers": [
                    {"type": "Conv2D", "size": "{{convSize}}", "strides": 2},
                    {"type": "Conv2D", "size": "{{convSize * 2}}", "strides": 2}
                ]}
            ],
            "layers": [
                {"template": "template1", "convSize": 64},
                {"template": "template1", "convSize": 128},
                {"type": "Conv2D", "size": 64, "strides": 2},
            ]
        }
        mrbuilder.build_register_model(model_definition)
        model_builder = mrbuilder.get_model(model_definition["name"])
        model = model_builder(self.input_shape)
        layers = model.layers

        self.assertEqual(len(layers),
                         6,
                         "number of layers is not correct")

        self.assertEqual(layers[1].filters,
                         model_definition["layers"][0]["convSize"],
                         "template size is not correct")
        self.assertEqual(layers[2].filters,
                         model_definition["layers"][0]["convSize"] * 2,
                         "template size is not correct")
        self.assertEqual(layers[1].__class__.__name__,
                         layers[-1].__class__.__name__,
                         "wrong template type")

        self.assertEqual(layers[3].filters,
                         model_definition["layers"][1]["convSize"],
                         "template size is not correct")
        self.assertEqual(layers[4].filters,
                         model_definition["layers"][1]["convSize"] * 2,
                         "template size is not correct")
        self.assertEqual(layers[3].__class__.__name__,
                         layers[-1].__class__.__name__,
                         "wrong template type")

    def test_model_build_template_multi_layers_calc(self):
        model_definition = {
            "name": "testModelTemplateSimple",
            "properties": {"prop1": 64},
            "templates": [
                {"name": "template1", "layers": [
                    {"type": "Conv2D", "size": "{{convSize}}", "strides": 2},
                    {"type": "Conv2D", "size": "{{convSize * 2}}", "strides": 2}
                ]}
            ],
            "layers": [
                {"template": "template1", "convSize": "{{prop1}}"},
                {"template": "template1", "convSize": "{{prop1 * 2}}"},
                {"type": "Conv2D", "size": 64, "strides": 2}
            ]
        }
        mrbuilder.build_register_model(model_definition)
        model_builder = mrbuilder.get_model(model_definition["name"])
        model = model_builder(self.input_shape)
        layers = model.layers
        model.summary()
        self.assertEqual(len(layers),
                         6,
                         "number of layers is not correct")

        self.assertEqual(layers[1].filters,
                         model_definition["properties"]["prop1"],
                         "template size is not correct")
        self.assertEqual(layers[2].filters,
                         model_definition["properties"]["prop1"] * 2,
                         "template size is not correct")

        self.assertEqual(layers[3].filters,
                         model_definition["properties"]["prop1"] * 2,
                         "template size is not correct")
        self.assertEqual(layers[4].filters,
                         model_definition["properties"]["prop1"] * 4,
                         "template size is not correct")

        self.assertEqual(layers[1].__class__.__name__,
                         layers[-1].__class__.__name__,
                         "wrong template type")


if __name__ == '__main__':
    unittest.main()
