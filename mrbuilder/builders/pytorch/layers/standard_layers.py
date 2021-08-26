import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder_models import PyTorchBuilderLayer
from ..layer_registry import register_layer
import mrbuilder.utils as mrbu


@register_layer("Conv2d")
class Conv2dBuilderLayer(PyTorchBuilderLayer):
    def __init__(self, config=None, connection=None):
        super().__init__(config, connection)
        self.out_size = None

        self.padding = None
        self.forward_pad = None
        self.kernel_size = None
        self.strides = None

    def init_layer(self,
                   in_channels=None,
                   size=None,
                   kernel=3,
                   dilation=1,
                   strides=1,
                   groups=1,
                   padding=None,
                   bias=True):
        self.out_size = size
        self.kernel_size = mrbu.repeat_if_single(kernel)
        self.strides = mrbu.repeat_if_single(strides)

        self.forward_pad = self.calculate_forward_padding(padding)
        if self.forward_pad is None:
            if padding is None or padding == 'valid':
                self.padding = (0, 0)
            else:
                self.padding = mrbu.repeat_if_single(padding)

        args = {}
        if padding is not None and type(padding) != str:
            args["padding"] = padding

        return nn.Conv2d(in_channels=self.previous_size[0] if in_channels is None else in_channels,
                         out_channels=size,
                         kernel_size=kernel,
                         stride=strides,
                         groups=groups,
                         dilation=dilation,
                         bias=bias,
                         **args)

    def forward(self, x):
        if self.forward_pad is not None:
            x = F.pad(x, self.forward_pad)

        return super().forward(x)

    def get_output_size(self):
        padding_h = self.forward_pad[2] + self.forward_pad[3] if self.forward_pad is not None else (self.padding[0] * 2)
        padding_w = self.forward_pad[0] + self.forward_pad[1] if self.forward_pad is not None else (self.padding[1] * 2)

        return [self.out_size,
                self._get_dim_output_size(self.previous_size[1],
                                          self.kernel_size[0],
                                          padding_h,
                                          self.strides[0]),
                self._get_dim_output_size(self.previous_size[2],
                                          self.kernel_size[1],
                                          padding_w,
                                          self.strides[1])]

    @staticmethod
    def _get_dim_output_size(input_size, kernel_size, padding, stride):
        return math.floor((input_size - kernel_size + padding) // stride + 1)

    def calculate_forward_padding(self, padding):
        if padding == 'same':
            # padding_w = self.kernel_size[1]/2
            # padding_h = self.kernel_size[0]/2

            out_height = math.ceil(float(self.previous_size[1]) / float(self.strides[1]))
            out_width = math.ceil(float(self.previous_size[2]) / float(self.strides[0]))

            padding_h = ((out_height - 1) * self.strides[1] + self.kernel_size[1] - self.previous_size[1]) / 2
            padding_w = ((out_width - 1) * self.strides[0] + self.kernel_size[0] - self.previous_size[2]) / 2

            return [math.ceil(padding_w),
                    math.floor(padding_w),
                    math.ceil(padding_h),
                    math.floor(padding_h)]
        elif isinstance(padding, (list, tuple)) and len(padding) == 4:
            return padding
        elif isinstance(padding, str) and padding != 'valid':
            raise NotImplementedError(F"Padding Type not implemented: ${padding}")
        else:
            return None


@register_layer("Dropout")
class DropoutBuilderLayer(PyTorchBuilderLayer):
    def init_layer(self, rate=0.5):
        return nn.Dropout(rate)


@register_layer("Linear", "Dense")
class LinearBuilderLayer(PyTorchBuilderLayer):
    def __init__(self, config=None, connection=None):
        super().__init__(config, connection)
        self.out_size = None

    def init_layer(self, in_features=None, in_multiplier=1, size=None):
        self.out_size = size

        in_size = in_features
        if in_size is None:
            in_size = self.previous_size[0] * in_multiplier

        return nn.Linear(in_features=in_size,
                         out_features=size)

    def get_output_size(self):
        return [self.out_size, *self.previous_size[1:]]


@register_layer("BatchNorm2d", "BatchNorm", "BatchNormalization")
class BatchNorm2dBuilderLayer(PyTorchBuilderLayer):
    def init_layer(self, size=None):
        return nn.BatchNorm2d(num_features=self.previous_size[0] if size is None else size)


@register_layer("Flatten")
class FlattenBuilderLayer(PyTorchBuilderLayer):
    def init_layer(self, dimension=1):
        return nn.Flatten(dimension)

    def get_output_size(self):
        result = 1
        for dim in self.previous_size:
            if dim != 0:
                result = result * dim

        return [result]


@register_layer("Concat", "Concatenate", "cat")
class ConcatBuilderLayer(PyTorchBuilderLayer):
    def forward(self, x, axis=1):
        return torch.cat(x, axis)

    def calculate_output_size(self, axis=1):
        output_sizes = []
        for con in self.connection:
            output_size = con.get_output_size()
            if len(output_sizes) == 0:
                output_sizes = [*output_size]
            else:
                output_sizes[axis] += output_size[axis]

        return output_sizes


@register_layer("MaxPool2d", "maxpooling2d")
class MaxPool2dBuilderLayer(PyTorchBuilderLayer):
    def __init__(self, config=None, connection=None):
        super().__init__(config, connection)
        self.pool_size = None

    def init_layer(self, size=2, strides=None):  # todo: add padding
        self.pool_size = size
        return nn.MaxPool2d(kernel_size=size, stride=strides)

    def get_output_size(self):
        return [self.previous_size[0], *[item // self.pool_size for item in self.previous_size[1:]]]


@register_layer("AdaptiveAvgPool2d", "GlobalAveragePooling2D")
class AdaptiveAvgPool2dBuilderLayer(PyTorchBuilderLayer):
    def __init__(self, config=None, connection=None):
        super().__init__(config, connection)
        self.pool_size = None

    def init_layer(self, size=None):
        self.pool_size = mrbu.repeat_if_single(size)
        return nn.AdaptiveAvgPool2d(self.pool_size)

    def forward(self, x, dimension=1, flatten=True):
        x = self.layer(x)
        return torch.flatten(x, dimension) if flatten else x

    def calculate_output_size(self, flatten=True):
        output_size = self.previous_size[0]
        for size in self.pool_size:
            output_size = output_size * size

        # todo: handle dimensions / not flatten
        # if flatten:
        #     output_size = mul(output_size)
        return [output_size]


@register_layer("Activation")
class ActivationBuilderLayer(PyTorchBuilderLayer):
    def get_activation_fn_key(self):
        return "function"


@register_layer("ReLU")
class ReLUBuilderLayer(PyTorchBuilderLayer):
    def init_layer(self):
        return nn.ReLU(inplace=True)


@register_layer("Softmax")
class SoftmaxBuilderLayer(PyTorchBuilderLayer):
    def init_layer(self):
        return nn.Softmax()


@register_layer("LeakyRelu")
class LeakyReLUBuilderLayer(PyTorchBuilderLayer):
    def init_layer(self):
        return nn.LeakyReLU()


@register_layer("LSTM")
class LSTMBuilderLayer(PyTorchBuilderLayer):
    def init_layer(self, in_size=None, size=None):
        return nn.LSTM(input_size=self.previous_size[0] if in_size is None else in_size,
                       hidden_size=size)

    def forward(self, x, more=False):
        if more:
            x = self.layer(x)
            x = x[:, -1, :]
            return self.do_activation(x)
        else:
            return super().forward(x)


@register_layer("View", "Reshape")
class ViewBuilderLayer(PyTorchBuilderLayer):
    def __init__(self, config=None, connection=None):
        super().__init__(config, connection)
        self.shape = None

    def init_layer(self, shape=None):
        super().init_layer()
        self.shape = shape

    def forward(self, x, shape=None):
        return x.view(-1, *shape)

    def get_output_size(self):
        return self.shape


@register_layer("Roll")
class RollBuilderLayer(PyTorchBuilderLayer):
    def forward(self, x, shift=0, axis=0):
        return torch.roll(x, shifts=shift, dims=axis)


@register_layer("SizeOf", "Print", "Debug")
class SizeOfBuilderLayer(PyTorchBuilderLayer):
    def forward(self, x):
        print(self.name, "is size:", x.size(), "previous size", self.previous_size)
        return x

# ATTENTION -------------------


# standard_layers = {
# "Conv2DTranspose": lambda config, connection:
#     nn.ConvTranspose2d(
#         filters=config("size"),
#         kernel_size=config("kernel", 3),
#         strides=config("strides", 2),
#         padding=config("padding", "same"),
#         activation=config("activation"),
#         kernel_initializer=config("kernelInitializer", "glorot_uniform")
#     )(connection),
# "UpSampling2D": lambda config, connection:
#     UpSampling2D(config("size", 2))(connection),
# "Reshape": lambda config, connection:
#     Reshape(target_shape=config("shape"))(connection),
# "Add": lambda config, connection:
#     Add()(connection),
# "LSTM": lambda config, connection:
#     LSTM(
#         units=config("size"),
#         return_sequences=config("returnSequences", False) or
#                          config("convertToEmbedding", False) or
#                          config("hasMore", False)
#     )(connection)
# }
