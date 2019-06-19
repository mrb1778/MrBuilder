from typing import Dict, Callable


class BuilderConfig:
    layers_builders: Dict[str, Callable]
    layer_option_builders: Dict[str, Callable]

    def __init__(self) -> None:
        super().__init__()
        self.layers_builders = {}
        self.layer_option_builders = {}

    def get_model_creator(self) -> Callable:
        pass

    def get_model_input_builder(self) -> Callable:
        pass

    def get_layer_builder(self, layer_type: str) -> Callable:
        return self.layers_builders[layer_type]

    def register_layer_builder(self, layer_type: str, layer_builder: Callable) -> None:
        self.layers_builders[layer_type] = layer_builder

    def has_layer_builder(self, layer_type: str) -> bool:
        return layer_type in self.layers_builders

    def get_layer_options_builder(self, name: str) -> Callable:
        return self.layer_option_builders[name]

    def register_layer_options_builder(self, name: str, layer_option_builder: Callable) -> None:
        self.layer_option_builders[name] = layer_option_builder

    def has_layer_options_builder(self, name: str) -> bool:
        return name in self.layer_option_builders
