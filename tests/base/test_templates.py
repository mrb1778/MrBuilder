from base.test_bootstrap import TestBuilderBase
from mrbuilder.model_builder import MissingLayerTypeException
from mrbuilder.variable_registry import VariableNotFoundException


class TestBuilderTemplates:
    class Base(TestBuilderBase.Base):
        @classmethod
        def setUpClass(cls) -> None:
            super().setUpClass()
    
            cls.input_shape = [32, 32, 3]
            # cls.input_shape = 32
    
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
            self.get_builder().build(model_definition)
            model_builder = self.get_builder().get_model(model_definition["name"])
            model = model_builder(self.input_shape)
            
            self.assertEqual(self.get_stride(model, 1),
                             model_definition["templates"][0]["strides"],
                             "template field is not correct")
    
            self.assertEqual(self.get_type(model, 1),
                             self.get_type(model, 2),
                             "wrong template type")
    
        def test_model_template_missing(self):
            model_definition_fail = {
                "name": "testModelFail",
                "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
                "layers": [
                    {"type": "MISSING_LAYER_TYPE", "size": 64},
                    {"type": "Dropout", "rate": "{{initialDropoutRate}}"},
                    {"type": "ConvActBN", "size": 64},
                    {"type": "MaxPool2d", "size": 2},
    
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "Linear", "size": "{{outputSize}}"},
                    {"type": "Activation", "function": "softmax"}
                ]
            }
            with self.assertRaises(MissingLayerTypeException, msg="Layer type was missing, but not caught"):
                self.get_builder().build(model_definition_fail)
                model_builder = self.get_builder().get_model(model_definition_fail["name"])
                model_builder(self.input_shape, self.base_params)
    
        def test_template_var_missing(self):
            model_definition_fail = {
                "name": "testModelFail",
                "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
                "layers": [
                    {"type": "Conv2D", "size": "{{missingProperty}}"},
                    {"type": "Dropout", "rate": "{{initialDropoutRate}}"},
                    {"type": "ConvActBN", "size": 64},
                    {"type": "MaxPool2d", "size": 2},
    
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "Linear", "size": "{{outputSize}}"},
                    {"type": "Activation", "function": "softmax"}
                ]
            }
            with self.assertRaises(VariableNotFoundException, msg="Variable was missing, but not caught"):
                self.get_builder().build(model_definition_fail)
                model_builder = self.get_builder().get_model(model_definition_fail["name"])
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
            self.get_builder().build(model_definition)
            model_builder = self.get_builder().get_model(model_definition["name"])
            model = model_builder(self.input_shape)
                
            self.assertEqual(self.get_num_layers(model),
                             6,
                             "number of layers is not correct")
    
            self.assertEqual(self.get_size(model, 1),
                             model_definition["layers"][0]["convSize"],
                             "template size is not correct")
            self.assertEqual(self.get_size(model, 2),
                             model_definition["layers"][0]["convSize"] * 2,
                             "template size is not correct")
            self.assertEqual(self.get_type(model, 1),
                             self.get_type(model, -1),
                             "wrong template type")
    
            self.assertEqual(self.get_size(model, 3),
                             model_definition["layers"][1]["convSize"],
                             "template size is not correct")
            self.assertEqual(self.get_size(model, 4),
                             model_definition["layers"][1]["convSize"] * 2,
                             "template size is not correct")
            self.assertEqual(self.get_type(model, 3),
                             self.get_type(model, -1),
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
            self.get_builder().build(model_definition)
            model_builder = self.get_builder().get_model(model_definition["name"])
            model = model_builder(self.input_shape)
            self.assertEqual(self.get_num_layers(model),
                             6,
                             "number of layers is not correct")
    
            self.assertEqual(self.get_size(model, 1),
                             model_definition["properties"]["prop1"],
                             "template size is not correct")
            self.assertEqual(self.get_size(model, 2),
                             model_definition["properties"]["prop1"] * 2,
                             "template size is not correct")
    
            self.assertEqual(self.get_size(model, 3),
                             model_definition["properties"]["prop1"] * 2,
                             "template size is not correct")
            self.assertEqual(self.get_size(model, 4),
                             model_definition["properties"]["prop1"] * 4,
                             "template size is not correct")
    
            self.assertEqual(self.get_type(model, 1),
                             self.get_type(model, -1),
                             "wrong template type")
