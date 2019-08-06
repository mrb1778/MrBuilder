import unittest

import mrbuilder
import mrbuilder.impl.keras_builder


class BuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        mrbuilder.load_models_from_path('../models')

        cls.input_shape = [32, 32, 3]

        cls.base_params = {
            "outputSize": 50
        }

    def test_vgg16(self):
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

    def test_squeezenet(self):
        model_builder = mrbuilder.get_model("squeezenet")
        params = {
            **self.base_params,
            "initialConv": 64,
            "initialSqueeze": 16
        }
        model = model_builder(self.input_shape, params)
        layers = model.layers
        self.assertEqual(layers[1].output.shape[-1],
                         params["initialConv"],
                         "output shape is not correct")


if __name__ == '__main__':
    unittest.main()
