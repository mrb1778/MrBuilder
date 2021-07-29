from base.test_bootstrap import TestBuilderBase
from mrbuilder.model_builder import MissingLayerTypeException


class TestBuilder:
    class Base(TestBuilderBase.Base):
        @classmethod
        def setUpClass(cls) -> None:
            cls.model_definition = {
                "name": "vgg16",
                "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
                "layers": [
                    {"type": "ConvActBN", "size": 64},
                    {"type": "Dropout", "rate": "{{initialDropoutRate}}"},
                    {"type": "ConvActBN", "size": 64},
                    {"type": "MaxPooling2D", "size": 2},
    
                    {"type": "ConvActBN", "size": 128},
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "ConvActBN", "size": 128},
                    {"type": "MaxPooling2D", "size": 2},
    
                    {"type": "ConvActBN", "size": 256},
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "ConvActBN", "size": 256},
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "ConvActBN", "size": 256},
                    {"type": "MaxPooling2D", "size": 2},
    
                    {"type": "ConvActBN", "size": 512},
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "ConvActBN", "size": 512},
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "ConvActBN", "size": 512},
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "ConvActBN", "size": 512},
                    {"type": "MaxPooling2D", "size": 2},
                    {"type": "ConvActBN", "size": 512},
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "ConvActBN", "size": 512},
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "ConvActBN", "size": 512},
                    {"type": "MaxPooling2D", "size": 2},
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
    
                    {"type": "Flatten"},
                    {"type": "Dense", "size": 512},
                    {"type": "Activation", "function": "softmax"},
                    {"type": "BatchNormalization"},
    
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "Dense", "size": "{{outputSize}}"},
                    {"type": "Activation", "function": "softmax"}
                ]
            }
            cls.get_builder().build(cls.model_definition)
            cls.input_shape = [32, 32, 3]
    
            cls.base_params = {
                "outputSize": 50
            }
    
        def test_model_build_basic(self):
            model_builder = self.get_builder().get("vgg16")
            model = model_builder(self.input_shape, self.base_params)
            
            # self.assertEqual(self.get_num_layers(model),
            #                  65,
            #                  "number of layers is not correct")
            # noinspection PyTypeChecker
            # self.assertEqual(self.get_in_size(model, 0).as_list(),
            #                  [None] + self.input_shape,
            #                  "x shape is not correct")
            self.assertEqual(self.get_out_channels(model, -1),
                             self.base_params["outputSize"],
                             "output shape is not correct")
    
        def test_model_build_expression(self):
            model_definition = {
                "name": "testModelExpression",
                "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
                "layers": [
                    {"type": "Conv2D", "size": "{{64 * 2}}"},
                    {"type": "Dropout", "rate": "{{initialDropoutRate}}"},
                    {"type": "ConvActBN", "size": 64},
                    {"type": "MaxPooling2D", "size": 2},
    
                    {"type": "Dropout", "rate": "{{dropoutRate}}"},
                    {"type": "Dense", "size": "{{outputSize}}"},
                    {"type": "Activation", "function": "softmax"}
                ]
            }
            self.get_builder().build(model_definition)
            model_builder = self.get_builder().get(model_definition["name"])
            conv_size = 64
            model = model_builder(self.input_shape, {
                **self.base_params,
                'convSize': conv_size
            })
            self.assertEqual(self.get_out_channels(model, 1),
                             conv_size * 2,
                             "output shape is not correct")
    
        def test_model_layer_missing(self):
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
                self.get_builder().build(model_definition_fail)
                model_builder = self.get_builder().get(model_definition_fail["name"])
                model_builder(self.input_shape, self.base_params)

        def test_model_build_multiple(self):
            model_builder1 = self.get_builder().get("vgg16")
            model1 = model_builder1(self.input_shape, self.base_params)
            model2 = model_builder1(self.input_shape, self.base_params)
            self.assertEqual(self.get_num_layers(model1), self.get_num_layers(model2),
                             msg="Length of layers do not equal")

            model_builder2 = self.get_builder().get("vgg16")
            model2_1 = model_builder2(self.input_shape, self.base_params)
            self.assertEqual(self.get_num_layers(model1), self.get_num_layers(model2_1),
                             msg="Length of layers do not equal")
