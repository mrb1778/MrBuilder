import torch
import torch.nn as nn

from mrbuilder_pytorch.builder_models import PyTorchBuilderLayer
from mrbuilder_pytorch.layer_registry import register_layer
from mrbuilder_pytorch.layers.standard_layers import Conv2dBuilderLayer, BatchNorm2dBuilderLayer


@register_layer("ConvActBN")
class ConvActBNBuilderLayer(PyTorchBuilderLayer):
    def __init__(self, config=None, connection=None):
        super().__init__(config, connection)
        self.conv2d_layer = Conv2dBuilderLayer(config, connection)
        self.batch_norm_layer = BatchNorm2dBuilderLayer(config, self.conv2d_layer)

        activation_fn_name = self.get_activation_fn_name()
        self.conv2d_layer.config_activation_fn = activation_fn_name \
            if activation_fn_name is not None else "relu"

    def run_initialization(self):
        super_init = super().run_initialization()
        self.conv2d_layer.run_initialization()

        self.batch_norm_layer.run_initialization()
        return super_init

    def forward(self, x, padding=None):
        self.conv2d_layer.forward(x, padding)
        x = self.conv2d_layer.do_activation(x)
        x = self.batch_norm_layer.forward(x)
        return x
