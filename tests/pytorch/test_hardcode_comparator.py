import unittest

import math
import torch
from torch import nn
from torch.nn import functional as F

import mrbuilder_pytorch as mrb
from test_bootstrap import TestBootstrapPytorch


class TestHardcodeComparatorPytorch(unittest.TestCase, TestBootstrapPytorch.Base):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.use_dropout = False  # dropout currently has issues with reproducability
        # cls.model_definition = {
        #     "name": "vgg16",
        #     "properties": {"initialDropoutRate": 0.3, "dropoutRate": 0.4},
        #     "layers": [
        #         {"type": "ConvActBN", "size": 64},
        #         {"type": "Dropout", "rate": "{{initialDropoutRate}}"},
        #         {"type": "ConvActBN", "size": 64},
        #         {"type": "MaxPool2d", "size": 2},
        #
        #         {"type": "ConvActBN", "size": 128},
        #         {"type": "Dropout", "rate": "{{dropoutRate}}"},
        #         {"type": "ConvActBN", "size": 128},
        #         {"type": "MaxPool2d", "size": 2},
        #
        #         {"type": "ConvActBN", "size": 256},
        #         {"type": "Dropout", "rate": "{{dropoutRate}}"},
        #         {"type": "ConvActBN", "size": 256},
        #         {"type": "Dropout", "rate": "{{dropoutRate}}"},
        #         {"type": "ConvActBN", "size": 256},
        #         {"type": "MaxPool2d", "size": 2},
        #
        #         {"type": "ConvActBN", "size": 512},
        #         {"type": "Dropout", "rate": "{{dropoutRate}}"},
        #         {"type": "ConvActBN", "size": 512},
        #         {"type": "Dropout", "rate": "{{dropoutRate}}"},
        #         {"type": "ConvActBN", "size": 512},
        #         {"type": "Dropout", "rate": "{{dropoutRate}}"},
        #         {"type": "ConvActBN", "size": 512},
        #         {"type": "MaxPool2d", "size": 2},
        #         {"type": "ConvActBN", "size": 512},
        #         {"type": "Dropout", "rate": "{{dropoutRate}}"},
        #         {"type": "ConvActBN", "size": 512},
        #         {"type": "Dropout", "rate": "{{dropoutRate}}"},
        #         {"type": "ConvActBN", "size": 512},
        #         {"type": "MaxPool2d", "size": 2},
        #         {"type": "Dropout", "rate": "{{dropoutRate}}"},
        #
        #         {"type": "Flatten"},
        #         {"type": "Linear", "size": 512},
        #         {"type": "Activation", "function": "softmax"},
        #         {"type": "BatchNorm2d"},
        #
        #         {"type": "Dropout", "rate": "{{dropoutRate}}"},
        #         {"type": "Linear", "size": "{{outputSize}}"},
        #         {"type": "Activation", "function": "softmax"}
        #     ]
        # }
        # mrb.build(cls.model_definition)
        #
        #
        # cls.base_params = {
        #     "outputSize": 50
        # }
        cls.input_shape = 3  # [32, 32, 3]

    @staticmethod
    def _get_same_padding(x, layer):
        # padding_w = layer.kernel_size[1] / 2
        # padding_h = layer.kernel_size[0] / 2

        previous_size = x.shape
        out_height = math.ceil(float(previous_size[2]) / float(layer.stride[1]))
        out_width = math.ceil(float(previous_size[3]) / float(layer.stride[0]))

        padding_h = ((out_height - 1) * layer.stride[1] + layer.kernel_size[1] - previous_size[2]) / 2
        padding_w = ((out_width - 1) * layer.stride[0] + layer.kernel_size[0] - previous_size[3]) / 2

        return F.pad(x, [math.ceil(padding_w),
                         math.floor(padding_w),
                         math.ceil(padding_h),
                         math.floor(padding_h)])

    def test_basic(self):
        class TestNet(nn.Module):
            def __init__(self):
                super(TestNet, self).__init__()
                self.layers = nn.ModuleDict()

                self.layers['Conv2d'] = nn.Conv2d(1, 6, 3)
                self.layers['ReLU'] = nn.ReLU()
                self.layers['MaxPool2d'] = nn.MaxPool2d(2)

                self.layers['Conv2d_2'] = nn.Conv2d(6, 16, 3)
                self.layers['ReLU_2'] = nn.ReLU()
                self.layers['MaxPool2d_2'] = nn.MaxPool2d(2)

                self.layers['Flatten'] = nn.Flatten()
                self.layers['Linear'] = nn.Linear(16 * 6 * 6, 120)
                self.layers['ReLU_3'] = nn.ReLU()
                self.layers['Linear_2'] = nn.Linear(120, 84)
                self.layers['ReLU_4'] = nn.ReLU()
                self.layers['Linear_3'] = nn.Linear(84, 10)

            def forward(self, x):
                x = self.layers.Conv2d(x)
                x = self.layers.ReLU(x)
                x = self.layers.MaxPool2d(x)

                x = self.layers.Conv2d_2(x)
                x = self.layers.ReLU_2(x)
                x = self.layers.MaxPool2d_2(x)

                x = torch.flatten(x, 1)

                x = self.layers.Linear(x)
                x = self.layers.ReLU_3(x)

                x = self.layers.Linear_2(x)
                x = self.layers.ReLU_4(x)

                x = self.layers.Linear_3(x)
                return x

        mrb_net_builder = mrb.build({
            "name": "TestNet",
            "layers": [
                {"type": "Conv2d", "size": 6},
                {"type": "ReLU"},
                {"type": "MaxPool2d", "size": 2},

                {"type": "Conv2d", "size": 16},
                {"type": "ReLU"},
                {"type": "MaxPool2d", "size": 2},

                {"type": "Flatten"},

                {"type": "Linear", "size": 120},
                {"type": "ReLU"},

                {"type": "Linear", "size": 84},
                {"type": "ReLU"},

                {"type": "Linear", "size": 10}
            ]
        })

        input_size = (1, 32, 32)

        self.reset_seed()
        standard_net = TestNet()

        self.reset_seed()
        mrb_net = mrb_net_builder(input_size)

        self.compare_networks(standard_net, mrb_net, torch.randn(1, *input_size))

    def test_multi_build(self):
        mrb_net_builder = mrb.build({
            "name": "TestNet",
            "layers": [
                {"type": "Conv2d", "size": 6},
                {"type": "ReLU"},
                {"type": "MaxPool2d", "size": 2},

                {"type": "Conv2d", "size": 16},
                {"type": "ReLU"},
                {"type": "MaxPool2d", "size": 2},

                {"type": "Flatten"},

                {"type": "Linear", "size": 120},
                {"type": "ReLU"},

                {"type": "Linear", "size": 84},
                {"type": "ReLU"},

                {"type": "Linear", "size": 10}
            ]
        })

        input_size = (1, 32, 32)

        self.reset_seed()
        mrb_net_1 = mrb_net_builder(input_size)

        self.reset_seed()
        mrb_net_2 = mrb_net_builder(input_size)

        self.compare_networks(mrb_net_1, mrb_net_2, torch.randn(1, *input_size))

    def test_padding_net(self):
        _test_self = self

        class TestNet(nn.Module):
            def __init__(self):
                super(TestNet, self).__init__()
                self.layers = nn.ModuleDict()

                self.layers['Conv2d'] = nn.Conv2d(1, 32, 3, bias=True)
                nn.init.xavier_uniform_(self.layers['Conv2d'].weight.data)
                self.layers['Dropout'] = nn.Dropout(0.2)

                self.layers['Conv2d_2'] = nn.Conv2d(32, 64, (3, 3), bias=True)
                nn.init.xavier_uniform_(self.layers['Conv2d_2'].weight.data)
                self.layers['MaxPool2d'] = nn.MaxPool2d(2)
                self.layers['Dropout_2'] = nn.Dropout(0.3)

                self.layers['Flatten'] = nn.Flatten()
                self.layers['Linear'] = nn.Linear(14400, 128)
                self.layers['Dropout_3'] = nn.Dropout(0.3)

                self.layers['Linear_2'] = nn.Linear(128, 3)
                # self.layers['Softmax'] = nn.Softmax()

            def forward(self, x):
                x = self.layers.Conv2d(x)
                x = F.relu(x)
                x = self.layers.Dropout(x)

                print('b4', x.shape)
                x = _test_self._get_same_padding(x, self.layers.Conv2d_2)
                print('after', x.shape)
                x = self.layers.Conv2d_2(x)
                x = F.relu(x)
                x = self.layers.MaxPool2d(x)
                x = self.layers.Dropout_2(x)

                x = torch.flatten(x, 1)

                print(x.shape)
                x = self.layers.Linear(x)
                x = F.relu(x)
                x = self.layers.Dropout_3(x)
                x = self.layers.Linear_2(x)
                x = F.softmax(x)

                return x

        mrb_net_builder = mrb.build({
            "name": "TestNet",
            "layers": [
                {"type": "Conv2d", "size": 32, "bias": True,
                 "activation": "relu", "weights": "xavier_uniform", "padding": "valid"},
                {"type": "Dropout", "rate": 0.2},

                {"type": "Conv2d", "size": 64, "bias": True
                    , "activation": "relu", "weights": "xavier_uniform", "padding": "same"},
                {"type": "MaxPool2d", "size": 2},
                {"type": "Dropout", "rate": 0.3},

                {"type": "Flatten"},

                {"type": "Linear", "size": 128, "activation": "relu"},
                {"type": "Dropout", "rate": 0.3},

                {"type": "Linear", "size": 3, "activation": "softmax"}
            ]
        })

        input_size = (1, 32, 32)

        self.reset_seed()
        standard_net = TestNet()
        # standard_net = mrb_net_builder(input_size)

        self.reset_seed()
        mrb_net = mrb_net_builder(input_size)
        # mrb_net = TestNet()

        self.compare_networks(standard_net, mrb_net, torch.randn(1, *input_size))

    def test_model_build_vgg16(self):
        _test_self = self

        class VGG16(nn.Module):
            def __init__(self, initial_conv_size=64, kernel_size=3, max_pool_size=2, padding=1, output_size=10):
                super(VGG16, self).__init__()
                self.layers = nn.ModuleDict()

                # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding,
                self.layers["conv1_1"] = nn.Conv2d(3, initial_conv_size, kernel_size=kernel_size, padding=padding)
                self.layers["relu1_1"] = nn.ReLU()
                self.layers["conv1_2"] = nn.Conv2d(initial_conv_size, initial_conv_size, kernel_size=kernel_size,
                                                   padding=padding)
                self.layers["relu1_2"] = nn.ReLU()
                self.layers["pool1"] = nn.MaxPool2d(max_pool_size)

                self.layers["conv2_1"] = nn.Conv2d(initial_conv_size, initial_conv_size * 2, kernel_size=kernel_size,
                                                   padding=padding)
                self.layers["relu2_1"] = nn.ReLU()
                self.layers["conv2_2"] = nn.Conv2d(initial_conv_size * 2, initial_conv_size * 2,
                                                   kernel_size=kernel_size, padding=padding)
                self.layers["relu2_2"] = nn.ReLU()
                self.layers["pool2"] = nn.MaxPool2d(max_pool_size)

                self.layers["conv3_1"] = nn.Conv2d(initial_conv_size * 2, initial_conv_size * 4,
                                                   kernel_size=kernel_size, padding=padding)
                self.layers["relu3_1"] = nn.ReLU()
                self.layers["conv3_2"] = nn.Conv2d(initial_conv_size * 4, initial_conv_size * 4,
                                                   kernel_size=kernel_size, padding=padding)
                self.layers["relu3_2"] = nn.ReLU()
                self.layers["conv3_3"] = nn.Conv2d(initial_conv_size * 4, initial_conv_size * 4,
                                                   kernel_size=kernel_size, padding=padding)
                self.layers["relu3_3"] = nn.ReLU()
                self.layers["pool3"] = nn.MaxPool2d(max_pool_size)

                self.layers["conv4_1"] = nn.Conv2d(initial_conv_size * 4, initial_conv_size * 8,
                                                   kernel_size=kernel_size, padding=padding)
                self.layers["relu4_1"] = nn.ReLU()
                self.layers["conv4_2"] = nn.Conv2d(initial_conv_size * 8, initial_conv_size * 8,
                                                   kernel_size=kernel_size, padding=padding)
                self.layers["relu4_2"] = nn.ReLU()
                self.layers["conv4_3"] = nn.Conv2d(initial_conv_size * 8, initial_conv_size * 8,
                                                   kernel_size=kernel_size, padding=padding)
                self.layers["relu4_3"] = nn.ReLU()
                self.layers["pool4"] = nn.MaxPool2d(max_pool_size)

                self.layers["conv5_1"] = nn.Conv2d(initial_conv_size * 8, initial_conv_size * 8,
                                                   kernel_size=kernel_size, padding=padding)
                self.layers["relu5_1"] = nn.ReLU()
                self.layers["conv5_2"] = nn.Conv2d(initial_conv_size * 8, initial_conv_size * 8,
                                                   kernel_size=kernel_size, padding=padding)
                self.layers["relu5_2"] = nn.ReLU()
                self.layers["conv5_3"] = nn.Conv2d(initial_conv_size * 8, initial_conv_size * 8,
                                                   kernel_size=kernel_size, padding=padding)
                self.layers["relu5_3"] = nn.ReLU()
                self.layers["pool5"] = nn.MaxPool2d(2, 2)

                # fully connected layers
                self.layers["avgpool"] = nn.AdaptiveAvgPool2d((7, 7))
                self.layers["flatten"] = nn.Flatten()

                self.layers["fc1"] = nn.Linear(7 * 7 * initial_conv_size * 8, 4096)
                self.layers["relu6_1"] = nn.ReLU()

                if _test_self.use_dropout:
                    self.layers["dropout1"] = nn.Dropout()

                self.layers["fc2"] = nn.Linear(4096, initial_conv_size * 64)
                self.layers["relu6_2"] = nn.ReLU()

                if _test_self.use_dropout:
                    self.layers["dropout2"] = nn.Dropout()

                self.layers["fc3"] = nn.Linear(initial_conv_size * 64, output_size)

            def forward(self, x):
                x = self.layers.conv1_1(x)
                x = self.layers.relu1_1(x)
                x = self.layers.conv1_2(x)
                x = self.layers.relu1_2(x)
                x = self.layers.pool1(x)

                x = self.layers.conv2_1(x)
                x = self.layers.relu2_1(x)
                x = self.layers.conv2_2(x)
                x = self.layers.relu2_2(x)
                x = self.layers.pool2(x)

                x = self.layers.conv3_1(x)
                x = self.layers.relu3_1(x)
                x = self.layers.conv3_2(x)
                x = self.layers.relu3_2(x)
                x = self.layers.conv3_3(x)
                x = self.layers.relu3_3(x)
                x = self.layers.pool3(x)

                x = self.layers.conv4_1(x)
                x = self.layers.relu4_1(x)
                x = self.layers.conv4_2(x)
                x = self.layers.relu4_2(x)
                x = self.layers.conv4_3(x)
                x = self.layers.relu4_3(x)
                x = self.layers.pool4(x)

                x = self.layers.conv5_1(x)
                x = self.layers.relu5_1(x)
                x = self.layers.conv5_2(x)
                x = self.layers.relu5_2(x)
                x = self.layers.conv5_3(x)
                x = self.layers.relu5_3(x)
                x = self.layers.pool5(x)

                x = self.layers.avgpool(x)
                x = self.layers.flatten(x)

                x = self.layers.fc1(x)
                x = self.layers.relu6_1(x)

                if _test_self.use_dropout:
                    x = self.layers.dropout1(x)

                x = self.layers.fc2(x)
                x = self.layers.relu6_2(x)

                if _test_self.use_dropout:
                    x = self.layers.dropout2(x)

                x = self.layers.fc3(x)

                return x

        mrb_net_builder = mrb.build({
            "name": "VGG16",
            "properties": {"initialConvSize": 64},
            "layers": [
                {"name": "conv1_1", "type": "Conv2d", "size": "{{initialConvSize}}", "padding": "{{padding}}"},
                {"name": "relu1_1", "type": "ReLU"},
                {"name": "conv1_2", "type": "Conv2d", "size": "{{initialConvSize}}", "padding": "{{padding}}"},
                {"name": "relu1_2", "type": "ReLU"},
                {"name": "pool1", "type": "MaxPool2d", "size": "{{maxPoolSize}}"},

                {"name": "conv2_1", "type": "Conv2d", "size": "{{initialConvSize * 2}}", "padding": "{{padding}}"},
                {"name": "relu2_1", "type": "ReLU"},
                {"name": "conv2_2", "type": "Conv2d", "size": "{{initialConvSize * 2}}", "padding": "{{padding}}"},
                {"name": "relu2_2", "type": "ReLU"},
                {"name": "pool2", "type": "MaxPool2d", "size": "{{maxPoolSize}}"},

                {"name": "conv3_1", "type": "Conv2d", "size": "{{initialConvSize * 4}}", "padding": "{{padding}}"},
                {"name": "relu3_1", "type": "ReLU"},
                {"name": "conv3_2", "type": "Conv2d", "size": "{{initialConvSize * 4}}", "padding": "{{padding}}"},
                {"name": "relu3_2", "type": "ReLU"},
                {"name": "conv3_3", "type": "Conv2d", "size": "{{initialConvSize * 4}}", "padding": "{{padding}}"},
                {"name": "relu3_3", "type": "ReLU"},
                {"name": "pool3", "type": "MaxPool2d", "size": "{{maxPoolSize}}"},
                {"name": "conv4_1", "type": "Conv2d", "size": "{{initialConvSize * 8}}", "padding": "{{padding}}"},
                {"name": "relu4_1", "type": "ReLU"},
                {"name": "conv4_2", "type": "Conv2d", "size": "{{initialConvSize * 8}}", "padding": "{{padding}}"},
                {"name": "relu4_2", "type": "ReLU"},
                {"name": "conv4_3", "type": "Conv2d", "size": "{{initialConvSize * 8}}", "padding": "{{padding}}"},
                {"name": "relu4_3", "type": "ReLU"},
                {"name": "pool4", "type": "MaxPool2d", "size": "{{maxPoolSize}}"},
                {"name": "conv5_1", "type": "Conv2d", "size": "{{initialConvSize * 8}}", "padding": "{{padding}}"},
                {"name": "relu5_1", "type": "ReLU"},
                {"name": "conv5_2", "type": "Conv2d", "size": "{{initialConvSize * 8}}", "padding": "{{padding}}"},
                {"name": "relu5_2", "type": "ReLU"},
                {"name": "conv5_3", "type": "Conv2d", "size": "{{initialConvSize * 8}}", "padding": "{{padding}}"},
                {"name": "relu5_3", "type": "ReLU"},
                {"name": "pool5", "type": "MaxPool2d", "size": "{{maxPoolSize}}"},
                {"name": "avgpool", "type": "AdaptiveAvgPool2d", "size": (7, 7)},
                {"name": "flatten", "type": "Flatten"},
                {"name": "fc1", "type": "Linear", "size": "{{initialConvSize * 64}}"},
                {"name": "relu6_1", "type": "ReLU"},
                {"name": "dropout1", "type": "Dropout", "if": self.use_dropout},

                {"name": "fc2", "type": "Linear", "size": "{{initialConvSize * 64}}"},
                {"name": "relu6_2", "type": "ReLU"},
                {"name": "dropout2", "type": "Dropout", "if": self.use_dropout},

                {"name": "fc3", "type": "Linear", "size": "{{output_size}}"},

            ]
        })

        input_size = (3, 32, 32)
        net_input = torch.randn(1, *input_size)
        params = {
            "initial_conv_size": 64,
            "kernel_size": 3,
            "max_pool_size": 2,
            "padding": 1,
            "output_size": 10
        }

        self.reset_seed()
        standard_net = VGG16(**params)

        self.reset_seed()
        mrb_net = mrb_net_builder(input_size, **params)

        self.compare_networks(standard_net, mrb_net, net_input)

        try:
            # noinspection PyPackageRequirements
            from torchvision import models

            torch_model = models.vgg16()
            self.compare_network_parameters(standard_net, torch_model)
            self.compare_network_parameters(mrb_net, torch_model)
        except ImportError as e:
            print("torchvision not installed, skipping vgg16 comparison")

    def test_same_padding_vs_tf(self):
        try:
            # noinspection PyPackageRequirements
            import tensorflow as tf

            input_size = (120, 120)
            kernel_size = 6
            _test_self = self

            def test_tf():

                def comp_conv2d(conv2d, X):
                    # Here (1, 1) indicates that the batch size and the number of channels
                    # are both 1
                    X = tf.reshape(X, (1,) + X.shape + (1,))
                    Y = conv2d(X)
                    # Exclude the first two dimensions that do not interest us: examples and
                    # channels
                    return tf.reshape(Y, Y.shape[1:3])

                # Note that here 1 row or column is padded on either side, so a total of 2
                # rows or columns are added
                conv2d = tf.keras.layers.Conv2D(1, kernel_size=kernel_size, padding='same')
                X = tf.random.uniform(shape=input_size)
                return comp_conv2d(conv2d, X).shape

            def test_pyt():
                def comp_conv2d(conv2d, X):
                    # Here (1, 1) indicates that the batch size and the number of channels
                    # are both 1
                    X = X.reshape((1, 1) + X.shape)
                    X = _test_self._get_same_padding(X, conv2d)
                    Y = conv2d(X)
                    # Exclude the first two dimensions that do not interest us: examples and
                    # channels
                    return Y.reshape(Y.shape[2:])

                # Note that here 1 row or column is padded on either side, so a total of 2
                # rows or columns are added
                conv2d = nn.Conv2d(1, 1, kernel_size=kernel_size)
                X = torch.rand(size=input_size)
                return comp_conv2d(conv2d, X).shape

            self.assertEqual(str(test_pyt()).replace('[', '').replace(']', '').split("(", 1)[1],
                             str(test_tf()).split("(", 1)[1], "not equal")
        except ImportError as e:
            print("Tensorflow not installed, skipping")

    def test_squeezenet(self):
        _test_self = self

        class SqueezeNet(nn.Module):
            def __init__(self):
                super(SqueezeNet, self).__init__()
                self.layers = nn.ModuleDict()

                self.layers['Conv2d'] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))
                self.layers['ReLU'] = nn.ReLU()
                self.layers['MaxPool2d'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

                self.layers['squeezed'] = nn.Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
                self.layers['left'] = nn.Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
                self.layers['right'] = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1))

                self.layers['squeezed_2'] = nn.Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
                self.layers['left_2'] = nn.Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
                self.layers['right_2'] = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1))

                self.layers['MaxPool2d_2'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                                                          ceil_mode=False)

                self.layers['squeezed_3'] = nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
                self.layers['left_3'] = nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
                self.layers['right_3'] = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1))

                self.layers['squeezed_4'] = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
                self.layers['left_4'] = nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
                self.layers['right_4'] = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1))

                self.layers['MaxPool2d_3'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1,
                                                          ceil_mode=False)

                self.layers['squeezed_5'] = nn.Conv2d(128, 48, kernel_size=(1, 1), stride=(1, 1))
                self.layers['left_5'] = nn.Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
                self.layers['right_5'] = nn.Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1))

                self.layers['squeezed_6'] = nn.Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1))
                self.layers['left_6'] = nn.Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
                self.layers['right_6'] = nn.Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1))

                self.layers['squeezed_7'] = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
                self.layers['left_7'] = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
                self.layers['right_7'] = nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1))

                self.layers['squeezed_8'] = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
                self.layers['left_8'] = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
                self.layers['right_8'] = nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1))
                if _test_self.use_dropout:
                    self.layers['Dropout'] = nn.Dropout(p=0.5, inplace=False)
                self.layers['Conv2d_2'] = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
                self.layers['ReLU_2'] = nn.ReLU()
                self.layers['AdaptiveAvgPool2d'] = nn.AdaptiveAvgPool2d(output_size=[3, 3])
                self.layers['Softmax'] = nn.Softmax()

            def forward(self, x):
                x = self.layers.Conv2d(x)
                x = self.layers.ReLU(x)
                x = self.layers.MaxPool2d(x)

                x = self.forward_squeeze(x)
                x = self.forward_squeeze(x, 2)
                x = self.layers.MaxPool2d_2(x)

                x = self.forward_squeeze(x, 3)
                x = self.forward_squeeze(x, 4)
                x = self.layers.MaxPool2d_3(x)

                x = self.forward_squeeze(x, 5)
                x = self.forward_squeeze(x, 6)

                x = self.forward_squeeze(x, 7)
                x = self.forward_squeeze(x, 8)

                if _test_self.use_dropout:
                    x = self.layers.Dropout(x)
                x = self.layers.Conv2d_2(x)
                x = self.layers.ReLU_2(x)
                x = self.layers.AdaptiveAvgPool2d(x)
                x = torch.flatten(x, 1)
                x = self.layers.Softmax(x)

                return x

            def forward_squeeze(self, x, postfix=None):
                postfix = '' if postfix is None else '_' + str(postfix)

                squeeze = self.layers['squeezed' + postfix](x)
                left = self.layers['left' + postfix](squeeze)

                squeeze = _test_self._get_same_padding(squeeze, self.layers['right' + postfix])
                right = self.layers['right' + postfix](squeeze)

                return torch.cat([left, right], axis=0)

        mrb_net_builder = mrb.build({
            "name": "SqueezeNet",
            "properties": {"initialConv": 64, "initialSqueeze": 16, "dropoutRate": 0.5},
            "templates": [
                {"name": "SqueezeExpand", "layers": [
                    {"type": "Conv2D", "size": "{{squeeze}}", "kernel": 1, "activation": "relu",
                     "name": "squeezed"},
                    {"type": "Conv2D", "size": "{{squeeze * 4}}", "kernel": 1, "activation": "relu",
                     # "padding": "same",
                     "name": "left", "connectionTo": "squeezed"},
                    {"type": "Conv2D", "size": "{{squeeze * 4}}", "kernel": 3, "activation": "relu", "padding": "same",
                     "name": "right", "connectionTo": "squeezed"},
                    {"type": "Concatenate", "connectionTo": ["left", "right"]}
                ]}
            ],
            "layers": [
                {"type": "Conv2D", "size": "{{initialConv}}", "kernel": 3, "strides": 2, "padding": "valid"},
                {"type": "ReLU"},
                {"type": "MaxPooling2D", "size": 3, "strides": 2},

                {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze}}"},
                {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze}}"},
                {"type": "MaxPooling2D", "size": 3, "strides": 2},

                {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 2}}"},
                {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 2}}"},
                {"type": "MaxPooling2D", "size": 3, "strides": 2},

                {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 3}}"},
                {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 3}}"},

                {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 4}}"},
                {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 4}}"},

                {"type": "Dropout", "rate": "{{dropoutRate}}", "if": self.use_dropout},

                {"type": "Conv2D", "size": "{{outputSize}}", "kernel": 1, "padding": "valid"},
                {"type": "ReLU"},
                {"type": "GlobalAveragePooling2D", "size": [3, 3], "strides": 2},
                {"type": "Softmax"}
            ]
        })

        input_size = (1, 128, 128)
        params = {"outputSize": 3}

        self.reset_seed()
        standard_net = SqueezeNet()

        self.reset_seed()
        mrb_net = mrb_net_builder(input_size, params)

        self.compare_networks(standard_net, mrb_net, torch.randn(1, *input_size))

    # def test_alex_net(self):
    #     class AlexNet(nn.Module):
    #         def __init__(self, num_classes=1000):
    #             super(AlexNet, self).__init__()
    #             self.layers = nn.ModuleDict({
    #                 "Conv2d_1": nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
    #                 # "BN_1":nn.BatchNorm2d(64),
    #                 "Act_1": nn.ReLU(inplace=True),
    #                 "Max_1": nn.MaxPool2d(kernel_size=3, stride=2),
    #                 "Conv2d_2": nn.Conv2d(64, 192, kernel_size=5, padding=2),
    #                 # "BN_2":nn.BatchNorm2d(192),
    #                 "Act_2": nn.ReLU(inplace=True),
    #                 "Max_2": nn.MaxPool2d(kernel_size=3, stride=2),
    #                 "Conv2d_3": nn.Conv2d(192, 384, kernel_size=3, padding=1),
    #                 # "BN_3":nn.BatchNorm2d(384),
    #                 "Act_3": nn.ReLU(inplace=True),
    #                 "Conv2d_4": nn.Conv2d(384, 256, kernel_size=3, padding=1),
    #                 # "BN_4":nn.BatchNorm2d(64),
    #                 "Act_4": nn.ReLU(inplace=True),
    #                 "Conv2d_5": nn.Conv2d(256, 256, kernel_size=3, padding=2),
    #                 # "BN_5":nn.BatchNorm2d(64),
    #                 "Act_5": nn.ReLU(inplace=True),
    #                 "Max_5": nn.MaxPool2d(kernel_size=3, stride=2),
    #                 "AvgPool": nn.AdaptiveAvgPool2d((6, 6)),
    #                 "Pool": nn.AdaptiveAvgPool2d((6, 6)),
    #                 "drop_6": nn.Dropout(),
    #                 "Linear_6": nn.Linear(256 * 6 * 6, 4096),
    #                 # "BN_6":nn.BatchNorm1d(4096),
    #                 "Act_6": nn.ReLU(inplace=True),
    #                 "drop_7": nn.Dropout(),
    #                 "Linear_7": nn.Linear(4096, 4096),
    #                 # "BN_7":nn.BatchNorm1d(4096),
    #                 "Act_7": nn.ReLU(inplace=True),
    #                 "Linear_8": nn.Linear(4096, num_classes),
    #                 # "BN_8":nn.BatchNorm1d(num_classes),
    #                 # "Softmax":nn.LogSoftmax()
    #             })
    #
    #         def forward(self, x):
    #             x = self.layers['Conv2d_1'](x)
    #             x = self.layers['Act_1'](x)
    #             x = self.layers['Max_1'](x)
    #             x = self.layers['Conv2d_2'](x)
    #             x = self.layers['Act_2'](x)
    #             x = self.layers['Max_2'](x)
    #             x = self.layers['Conv2d_3'](x)
    #             x = self.layers['Act_3'](x)
    #             x = self.layers['Conv2d_4'](x)
    #             x = self.layers['Act_4'](x)
    #             x = self.layers['Conv2d_5'](x)
    #             x = self.layers['Act_5'](x)
    #             x = self.layers['Max_5'](x)
    #             x = self.layers['AvgPool'](x)
    #             x = x.view(-1, 256 * 6 * 6)
    #             x = self.layers['Linear_6'](x)
    #             x = self.layers['Act_6'](x)
    #             x = self.layers['Linear_7'](x)
    #             x = self.layers['Act_7'](x)
    #             x = self.layers['Linear_8'](x)
    #             return x
    #
    #
    #     mrb_net_builder = mrb.build({
    #         "name": "AlexNet",
    #         "layers": [
    #             {"type": "Conv2D", "size": "{{initialConv}}", "kernel": 3, "strides": 2, "padding": "valid"},
    #             {"type": "ReLU"},
    #             {"type": "MaxPooling2D", "size": 3, "strides": 2},
    #
    #             {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze}}"},
    #             {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze}}"},
    #             {"type": "MaxPooling2D", "size": 3, "strides": 2},
    #
    #             {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 2}}"},
    #             {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 2}}"},
    #             {"type": "MaxPooling2D", "size": 3, "strides": 2},
    #
    #             {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 3}}"},
    #             {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 3}}"},
    #
    #             {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 4}}"},
    #             {"template": "SqueezeExpand", "squeeze": "{{initialSqueeze * 4}}"},
    #
    #             {"type": "Dropout", "rate": "{{dropoutRate}}"},
    #
    #             {"type": "Conv2D", "size": "{{outputSize}}", "kernel": 1, "padding": "valid"},
    #             {"type": "ReLU"},
    #             {"type": "GlobalAveragePooling2D", "size": [3, 3], "strides": 2},
    #             {"type": "Softmax"}
    #         ]
    #     })
    #
    #     input_size = (3, 32, 32)
    #
    #     self.reset_seed()
    #     standard_net = AlexNet()
    #     print(standard_net)
    #     self.reset_seed()
    #     # mrb_net = mrb_net_builder(input_size)
    #     mrb_net = mrb_net_builder(input_size)
    #
    #     self.compare_networks(standard_net, mrb_net, torch.randn(1, *input_size))


if __name__ == '__main__':
    unittest.main()
