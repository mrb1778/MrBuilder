import unittest


class TestBootstrapBase:
    @classmethod
    def get_builder(cls):
        return None

    @classmethod
    def get_num_layers(cls, model) -> int:
        raise NotImplemented

    @classmethod
    def get_layer(cls, model, i):
        raise NotImplemented

    @classmethod
    def get_type(cls, model, i):
        raise NotImplemented

    @classmethod
    def get_stride(cls, model, layer_num, dimension=0):
        raise NotImplemented

    @classmethod
    def get_size(cls, model, i):
        raise NotImplemented

    @classmethod
    def get_in_size(cls, model, i):
        raise NotImplemented

    @classmethod
    def get_out_size(cls, model, i):
        raise NotImplemented

    @classmethod
    def get_in_size(cls, model, i):
        raise NotImplemented

    @classmethod
    def get_out_size(cls, model, i):
        raise NotImplemented


class TestBuilderBase:
    class Base(TestBootstrapBase, unittest.TestCase):
        pass
