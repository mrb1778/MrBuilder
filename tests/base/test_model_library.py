from base.test_bootstrap import TestBuilderBase


class TestModelLibrary:
    class Base(TestBuilderBase.Base):
        @classmethod
        def setUpClass(cls) -> None:
            super().setUpClass()
            cls.get_builder().load()

            cls.input_shape = [32, 32, 3]

            cls.base_params = {
                "outputSize": 50
            }

        def test_vgg16(self):
            model_builder = self.get_builder().get_model("vgg16")
            model = model_builder(self.input_shape, self.base_params)
            self.assertEqual(self.get_num_layers(model),
                             65,
                             "number of layers is not correct")
            # noinspection PyTypeChecker
            self.assertEqual(self.get_in_size(model, 0).as_list(),
                             [None] + self.input_shape,
                             "x shape is not correct")
            self.assertEqual(self.get_out_size(model, -1)[1],
                             self.base_params["outputSize"],
                             "output shape is not correct")

        def test_squeezenet(self):
            model_builder = self.get_builder().get_model("squeezenet")
            params = {
                **self.base_params,
                "initialConv": 64,
                "initialSqueeze": 16
            }
            model = model_builder(self.input_shape, params)
            layers = model.layers
            self.assertEqual(self.get_out_size(model, 1)[-1],
                             params["initialConv"],
                             "output shape is not correct")