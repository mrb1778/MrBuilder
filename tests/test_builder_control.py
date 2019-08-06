import unittest

import mrbuilder
import mrbuilder.impl.keras_builder
from mrbuilder.model_builder import MissingLayerTypeException
from mrbuilder.variable_registry import VariableNotFoundException


class BuilderControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.input_shape = [32, 32, 3]

        cls.base_params = {
            "outputSize": 50
        }

    def test_model_build_with_if_keep(self):
        model_definition = {
            "name": "testModelTemplateSimple",
            "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
            "templates": [
                {"name": "template1", "type": "Conv2D", "strides": 2}
            ],
            "layers": [
                {"template": "template1", "size": 64},
                {"type": "Conv2D", "size": 64, "strides": 2, "if": True},
                {"type": "Conv2D", "size": 64, "strides": 2}
            ]
        }
        mrbuilder.build_register_model(model_definition)
        model_builder = mrbuilder.get_model(model_definition["name"])
        model = model_builder(self.input_shape)
        layers = model.layers

        self.assertEqual(4,
                         len(layers),
                         "number of layers is not correct")

        self.assertEqual(layers[1].strides[0],
                         model_definition["templates"][0]["strides"],
                         "template field is not correct")

        self.assertEqual(layers[1].__class__.__name__,
                         layers[2].__class__.__name__,
                         "wrong template type")

    def test_model_build_with_if_remove(self):
        model_definition = {
            "name": "testModelTemplateSimple",
            "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
            "templates": [
                {"name": "template1", "type": "Conv2D", "strides": 2}
            ],
            "layers": [
                {"template": "template1", "size": 64},
                {"type": "Conv2D", "size": 64, "strides": 2, "if": False},
                {"type": "Conv2D", "size": 64, "strides": 2}
            ]
        }
        mrbuilder.build_register_model(model_definition)
        model_builder = mrbuilder.get_model(model_definition["name"])
        model = model_builder(self.input_shape)
        layers = model.layers

        self.assertEqual(3,
                         len(layers),
                         "number of layers is not correct")

        self.assertEqual(layers[1].strides[0],
                         model_definition["templates"][0]["strides"],
                         "template field is not correct")

        self.assertEqual(layers[1].__class__.__name__,
                         layers[2].__class__.__name__,
                         "wrong template type")

    def test_model_build_with_if_with_expression(self):
        model_definition = {
            "name": "testModelTemplateSimple",
            "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4, "keep1St": True},
            "templates": [
                {"name": "template1", "type": "Conv2D", "strides": 2}
            ],
            "layers": [
                {"template": "template1", "size": 64, "if": "{{keep1St}}"},
                {"type": "Conv2D", "size": 64, "strides": 2, "if": "{{2 > 1}}"},
                {"type": "Conv2D", "size": 64, "strides": 2, "if": "{{2 < 1}}"},
                {"type": "Conv2D", "size": 64, "strides": 2, "if": "{{2 < 1}}"},
                {"type": "Conv2D", "size": 64, "strides": 2}
            ]
        }
        mrbuilder.build_register_model(model_definition)
        model_builder = mrbuilder.get_model(model_definition["name"])
        model = model_builder(self.input_shape)
        layers = model.layers

        self.assertEqual(4,
                         len(layers),
                         "number of layers is not correct")

        self.assertEqual(layers[1].strides[0],
                         model_definition["templates"][0]["strides"],
                         "template field is not correct")

        self.assertEqual(layers[1].__class__.__name__,
                         layers[2].__class__.__name__,
                         "wrong template type")


if __name__ == '__main__':
    unittest.main()
