import unittest

import mrbuilder
import mrbuilder.impl.keras_builder
from mrbuilder.builder_registry import MissingModelException


class BuilderRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.model_definition_name = "testModel"
        cls.model_definition = {
            "name": "testModel",
            "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
            "layers": [
                {"type": "ConvActBN", "size": 64},
                {"type": "Dropout", "rate": "{{initialDropoutRate}}"},
                {"type": "ConvActBN", "size": 64},
                {"type": "MaxPooling2D", "size": 2},

                {"type": "Dropout", "rate": "{{dropoutRate}}"},
                {"type": "Dense", "size": "{{outputSize}}"},
                {"type": "Activation", "function": "softmax"}
            ]
        }
        mrbuilder.build_register_model(cls.model_definition)

        cls.model_definition_name_custom = cls.model_definition_name + "Custom"
        mrbuilder.build_register_model(cls.model_definition, cls.model_definition_name_custom)

    def test_model_retrieval(self):
        self.assertIsNotNone(mrbuilder.get_model(self.model_definition_name),
                             "model not found")

    def test_model_retrieval_custom_name(self):
        self.assertIsNotNone(mrbuilder.get_model(self.model_definition_name_custom),
                             "can not retrieve model with custom name")

    def test_model_retrieval_missing(self):
        with self.assertRaises(MissingModelException,
                               msg="MissingModelException not thrown when trying to find a non existent model "):
            mrbuilder.get_model(self.model_definition_name + "_INVALID")


if __name__ == '__main__':
    unittest.main()
