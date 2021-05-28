from mrbuilder.layer_registry import register_layer_wrapper

registered_layers = {}


def register_layer(name: str, *aliases):
    return register_layer_wrapper(registered_layers, name, *aliases)
