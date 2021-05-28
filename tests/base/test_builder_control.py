from base.test_bootstrap import TestBuilderBase


class TestBuilderControl:
    class Base(TestBuilderBase.Base):
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
            self.get_builder().build(model_definition)
            model_builder = self.get_builder().get(model_definition["name"])
            model = model_builder(self.input_shape)

            self.assertEqual(4,
                             self.get_num_layers(model),
                             "number of layers is not correct")

            self.assertEqual(self.get_stride(model, 1),
                             model_definition["templates"][0]["strides"],
                             "template field is not correct")

            self.assertEqual(self.get_type(model, 1),
                             self.get_type(model, 2),
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
            self.get_builder().build(model_definition)
            model_builder = self.get_builder().get(model_definition["name"])
            model = model_builder(self.input_shape)
            layers = model.layers

            self.assertEqual(3,
                             self.get_num_layers(model),
                             "number of layers is not correct")

            self.assertEqual(self.get_stride(model, 1),
                             model_definition["templates"][0]["strides"],
                             "template field is not correct")

            self.assertEqual(self.get_type(model, 1),
                             self.get_type(model, 2),
                             "wrong template type")

        def test_model_build_with_if_with_expression(self):
            model_definition = {
                "name": "testModelTemplateSimple",
                "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4, "keep1St": True},
                "templates": [
                    {"name": "template1", "type": "Conv2D", "strides": 20}
                ],
                "layers": [
                    {"template": "template1", "size": 32, "if": "{{keep1St}}"},
                    {"type": "Conv2D", "size": 16, "strides": 2, "if": "{{2 > 1}}"},
                    {"type": "Conv2D", "size": 256, "strides": 17, "if": "{{2 < 1}}"},
                    {"type": "Conv2D", "size": 512, "strides": 30, "if": "{{2 < 1}}"},
                    {"type": "Conv2D", "size": 128, "strides": 2}
                ]
            }
            self.get_builder().build(model_definition)
            model_builder = self.get_builder().get(model_definition["name"])
            model = model_builder(self.input_shape)
            
            self.assertEqual(4,
                             self.get_num_layers(model),
                             "number of layers is not correct")

            self.assertEqual(self.get_stride(model, 1),
                             model_definition["templates"][0]["strides"],
                             "template field is not correct")

            self.assertEqual(self.get_type(model, 1),
                             self.get_type(model, 2),
                             "wrong template type")

        # def test_model_build_with_repeat_count(self):
        #     model_definition = {
        #         "name": "testModelTemplateSimple",
        #         "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
        #         "templates": [
        #             {"name": "template1", "type": "Conv2D", "strides": 2}
        #         ],
        #         "layers": [
        #             {"template": "template1", "size": 64},
        #             {"type": "Conv2D", "size": 64, "strides": 2, "repeat": 2, "repeat-count": "i"},
        #             {"type": "Conv2D", "size": 64, "strides": 2}
        #         ]
        #     }
        #     self.get_builder().build(model_definition)
        #     model_builder = self.get_builder().get(model_definition["name"])
        #     model = model_builder(self.input_shape)
        #     layers = model.layers
        #
        #     self.assertEqual(5,
        #                      self.get_num_layers(model),
        #                      "number of layers is not correct")
        #
        #     self.assertEqual(self.get_strides(model, 1)[0],
        #                      model_definition["templates"][0]["strides"],
        #                      "template field is not correct")
        #
        #     self.assertEqual(self.get_type(model, 1),
        #                      self.get_type(model, 2),
        #                      "wrong template type")
