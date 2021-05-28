from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from mrbuilder.utils import get_params, remove_keys, is_single
from mrbuilder_pytorch import utils as pu


class PyTorchBuilderLayer:
    def __init__(self, config=None, connection=None):
        self.connection = connection
        self.name = self.__class__.__name__.replace("BuilderLayer", "")

        self.layer = None
        self.output = None
        self.input_shape = None

        self.config_params_init = {}
        self.config_params_forward = {}
        self.config_calculate_output_size = {}

        self.config_size = None
        self.config_activation_fn = None
        self.weight_init_fn = None

        self.previous_size = None

        if config is not None:
            self._populate_config_params(config)
            self.name = config("name", self.name)

    def _populate_config_params(self, config):
        activation_fn_name = self.get_activation_fn_name()
        self.config_activation_fn = activation_fn_name \
            if activation_fn_name is not None else config(self.get_activation_fn_key())

        self.weight_init_fn = config("weights")

        default_config_params_init = get_params(self.init_layer)
        # remove_keys(default_config_params_init, get_params(PyTorchBuilderLayer.init_layer).keys())
        self.config_params_init = {
            name: config(name, default_value) for name, default_value in default_config_params_init.items()
        }

        default_config_params_forward = get_params(self.forward)
        remove_keys(default_config_params_forward, get_params(PyTorchBuilderLayer.forward).keys())
        self.config_params_forward = {
            name: config(name, default_value) for name, default_value in default_config_params_forward.items()
        }

        default_config_get_output_size = get_params(self.calculate_output_size)
        self.config_calculate_output_size = {
            name: config(name, default_value) for name, default_value in default_config_get_output_size.items()
        }

    def run_initialization(self):
        if self.connection is not None:
            if is_single(self.connection):
                self.previous_size = self.connection.get_output_size()
            else:
                return [con.get_output_size() for con in self.connection]

        # noinspection PyArgumentList
        self.layer = self.init_layer(**self.config_params_init)
        self.init_weights()

    # noinspection PyMethodMayBeStatic
    def init_layer(self):
        return None

    def init_weights(self):
        if self.weight_init_fn and self.layer:
            if self.weight_init_fn == 'normal':
                nn.init.normal_(self.layer.weight.data)
            elif self.weight_init_fn in ['orth', "orthogonal"]:
                nn.init.orthogonal_(self.layer.weight.data)
            elif self.weight_init_fn in ['xavier',
                                         'xavierUniform',
                                         'xavier_uniform',
                                         'glorotUniform',
                                         'glorot_Uniform']:
                nn.init.xavier_uniform_(self.layer.weight.data)
            else:
                raise Exception(F"Weight Init {self.weight_init_fn} not implemented.")

    # noinspection PyShadowingBuiltins
    def run_forward(self, input=None):
        # todo: determine if better way to take care of network input. and null connection 
        expected_input = self.get_input()
        # noinspection PyArgumentList
        self.output = self.forward(expected_input if expected_input is not None else input,
                                   **self.config_params_forward)
        return self.output

    def forward(self, x):
        if self.layer is not None:
            output = self.layer(x)
            return self.do_activation(output)
        else:
            return self.do_activation(x)

    def do_activation(self, x):
        activation_fn = self.get_activation_fn()
        return activation_fn(x) if activation_fn is not None else x

    # noinspection PyMethodMayBeStatic
    def get_weight_init_fn_key(self):
        return "weights"

    def get_activation_fn_key(self):
        return "activation"

    # noinspection PyMethodMayBeStatic
    def get_activation_fn_name(self):
        return None

    def get_activation_fn(self):
        activation_fn_name = self.config_activation_fn
        if activation_fn_name is not None:
            return pu.get_activation_fn(activation_fn_name)
        else:
            return None

    def get_input(self):
        if self.connection is None:
            return None
        elif is_single(self.connection):
            return self.connection.output
        else:
            return [con.output for con in self.connection]

    # noinspection PyMethodMayBeStatic
    def get_output_size(self):
        return self.calculate_output_size(**self.config_calculate_output_size)

    def calculate_output_size(self, **kwargs):
        return self.previous_size

    # noinspection PyMethodMayBeStatic
    def get_ignored_fields(self):
        return 'connection'

    def get_modules(self, prefix=''):
        modules = {}
        for name, module in self.__dict__.items():
            if name not in self.get_ignored_fields():
                if isinstance(module, nn.Module):
                    modules[prefix + self.name + '.' + name] = module
                elif isinstance(module, PyTorchBuilderLayer):
                    child_modules = module.get_modules(prefix=prefix + '.' + name + '.')
                    if len(child_modules) > 0:
                        modules.update(child_modules)
        if len(modules) == 1:
            return {self.name: next(iter(modules.values()))}
        return modules


class PyTorchBuilderLayerInput(PyTorchBuilderLayer):
    def __init__(self, input_size=None, config=None, connection=None):
        super(PyTorchBuilderLayerInput, self).__init__(config, connection)
        self.input_size = input_size

    def get_output_size(self):
        return self.input_size


# class PyTorchBuilderLayerMulti(PyTorchBuilderLayer):
#     def __init__(self, config=None, connection=None):
#         super(PyTorchBuilderLayerMulti, self).__init__(config, connection)
#         self.layers = []


class PyTorchBuilderModel(nn.Module):
    def __init__(self, inputs, layers: List[PyTorchBuilderLayer], layers_by_name, output_config, name=None):
        super(PyTorchBuilderModel, self).__init__()

        self.name = name
        self.inputs = inputs
        self.builder_layers = layers

        self.output_config = output_config

        # layer_names
        layer_names = []
        pytorch_layers = {}
        for builder_layer in self.builder_layers:
            base_layer_name = builder_layer.name
            layer_name = base_layer_name
            index = 2
            while layer_name in layer_names:
                layer_name = base_layer_name + '_' + str(index)
                index = index + 1

            layer_names.append(layer_name)

            if layer_name != base_layer_name:
                builder_layer.name = layer_name

            builder_layer.run_initialization()
            modules = builder_layer.get_modules()
            pytorch_layers.update(modules)
        
        self.layers = nn.ModuleDict(pytorch_layers)

    def forward(self, x):
        out = None  # todo: determine if better way to get output
        for builder_layer in self.builder_layers:
            out = builder_layer.run_forward(x)

        return out

    def _get_name(self):
        return self.name \
            if self.name is not None \
            else self.__class__.__name__

    def predict(self, x):
        # todo: handle batch 1
        x = F.softmax(self.forward(x))
        return x

    def accuracy(self, x, y):
        prediction = self.predict(x)
        _, indices = torch.max(prediction, 1)
        acc = 100 * torch.sum(torch.eq(indices.float(), y.float()).float())/y.size()[0]
        return acc.cpu().data[0]
