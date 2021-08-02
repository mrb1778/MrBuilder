import io
import random
import torch

import mrbuilder.builders.pytorch as mrb
from base.test_bootstrap import TestBootstrapBase
import utils


class TestBootstrapPytorch:
    class Base(TestBootstrapBase):
        @classmethod
        def get_builder(cls):
            return mrb

        @classmethod
        def get_num_layers(cls, model):
            return len(model.builder_layers)

        @classmethod
        def get_layer(cls, model, i):
            return list(model.layers.values())[i-1]

        @classmethod
        def get_type(cls, model, i):
            return cls.get_layer(model, i).__class__.__name__

        @classmethod
        def get_stride(cls, model, layer_num, dimension=0):
            strides = cls.get_layer(model, layer_num).stride
            return strides[0] if isinstance(strides, (list, tuple)) else strides

        @classmethod
        def get_size(cls, model, i):
            return cls.get_layer(model, i).out_channels

        @classmethod
        def get_in_channels(cls, model, i):
            raise model.builder_layers[i].get_input_size()[0]

        @classmethod
        def get_out_channels(cls, model, i):
            return model.builder_layers[i].get_output_size()[0]

        #  END TestBootstrapBase

        @staticmethod
        def reset_seed():
            seed = 1778
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        @classmethod
        def net_summary(cls, net) -> str:
            print_buffer = io.StringIO()
            print(net, file=print_buffer)
            return print_buffer.getvalue()

        def compare_networks(self, n1, n2, net_input):
            self.compare_network_summaries(n1, n2)
            self.compare_network_parameters(n1, n2)
            self.compare_network_outputs(n1, n2, net_input)

        def compare_network_summaries(self, n1, n2):
            # noinspection PyUnresolvedReferences
            self.assertEqual(utils.model_summary(n1),
                             utils.model_summary(n2),
                             "summaries do not equal")

        def compare_network_outputs(self, n1, n2, net_input):
            self.reset_seed()
            n1_output = n1(net_input)
            self.reset_seed()
            n2_output = n2(net_input)

            # noinspection PyUnresolvedReferences
            self.assertEqual(len(n1_output),
                             len(n2_output),
                             "length of outputs are not equal")
            # noinspection PyUnresolvedReferences
            self.assertEqual(n1_output.shape,
                             n2_output.shape,
                             "shape of outputs are not equal")
            # noinspection PyUnresolvedReferences
            self.assertTrue(torch.equal(n1_output,
                                        n2_output),
                            f"outputs do not match \n{n1_output}\n{n2_output}")

        def compare_network_parameters(self, n1, n2):
            n1_params = list(n1.parameters())
            n2_params = list(n2.parameters())

            # noinspection PyUnresolvedReferences
            self.assertEqual(len(n1_params),
                             len(n2_params),
                             "length of params are not equal")
