from ..builder_models import PyTorchBuilderLayer
from ..layer_registry import register_layer
from ..layers.standard_layers import Conv2dBuilderLayer, BatchNorm2dBuilderLayer


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


@register_layer("SeparableConv2d")
class SeparableConv2d(PyTorchBuilderLayer):
    def __init__(self, config=None, connection=None):
        super().__init__(config, connection)

        self.depth_wise_conv2d = Conv2dBuilderLayer(config, connection)
        self.point_wise_conv2d = Conv2dBuilderLayer(config, self.depth_wise_conv2d)

    def run_initialization(self,
                           in_channels=None,
                           size=None,
                           kernel=3,
                           bias=True):
        super_init = super().run_initialization()

        in_channels = self.previous_size[0] if in_channels is None else in_channels
        self.depth_wise_conv2d.run_initialization(in_channels=in_channels,
                                                  size=in_channels,
                                                  kernel=kernel,
                                                  groups=in_channels,
                                                  padding=1,
                                                  bias=bias)
        self.point_wise_conv2d.run_initialization(in_channels=in_channels,
                                                  size=size,
                                                  kernel=1,
                                                  bias=bias)
        return super_init

    def forward(self, x):
        out = self.depth_wise_conv2d.forward(x)
        out = self.point_wise_conv2d.forward(out)
        return out
