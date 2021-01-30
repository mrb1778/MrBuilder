import unittest

from mrbuilder.builder_registry import MissingModelException


class TestBuilderRegistry:
    class Base(unittest.TestCase):
        @classmethod
        def get_builder(cls):
            return None

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
                    {"type": "MaxPool2d", "size": 2},

                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "Linear", "size": "{{outputSize}}"},
                    {"type": "Activation", "function": "softmax"}
                ]
            }
            cls.get_builder().build(cls.model_definition)

            cls.model_definition_name_custom = cls.model_definition_name + "Custom"
            cls.get_builder().build(cls.model_definition, cls.model_definition_name_custom)

        def test_model_retrieval(self):
            self.assertIsNotNone(self.get_builder().get_model(self.model_definition_name),
                                 "model not found")

        def test_model_retrieval_custom_name(self):
            self.assertIsNotNone(self.get_builder().get_model(self.model_definition_name_custom),
                                 "can not retrieve model with custom name")

        def test_model_retrieval_missing(self):
            with self.assertRaises(MissingModelException,
                                   msg="MissingModelException not thrown when trying to find a non existent model "):
                self.get_builder().get_model(self.model_definition_name + "_INVALID")