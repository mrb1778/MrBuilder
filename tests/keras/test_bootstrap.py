import mrbuilder_keras as mrb
from base.test_bootstrap import TestBootstrapBase


class TestBootstrapKeras(TestBootstrapBase):
    @classmethod
    def get_builder(cls):
        return mrb

    @classmethod
    def get_num_layers(cls, model):
        return len(model.layers)

    @classmethod
    def get_layer(cls, model, i):
        return None

    @classmethod
    def get_type(cls, model, i):
        return model.layers[i].__class__.__name__

    @classmethod
    def get_stride(cls, model, layer_num, dimension=0):
        strides = model.layers[layer_num].strides
        return strides[0] if isinstance(strides, (list, tuple)) else strides

    @classmethod
    def get_size(cls, model, i):
        return model.layers[i].filters

    @classmethod
    def get_in_size(cls, model, i):
        return model.layers[i].input.shape

    @classmethod
    def get_out_size(cls, model, i):
        return model.layers[i].output.shape

