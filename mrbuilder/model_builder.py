from typing import Callable, Dict

from mrbuilder.builder_config import BuilderConfig
from mrbuilder.builder_registry import BuilderRegistry


class ModelBuilder:
    builder_registry: BuilderRegistry
    builder_config: BuilderConfig

    layers: list
    layers_by_name: Dict
    properties: Dict

    def __init__(self, builder_registry, builder_config: BuilderConfig, model_config: Dict) -> None:
        super().__init__()
        self.builder_registry = builder_registry
        self.builder_config = builder_config

        self.model_config = model_config
        self.layers = []
        self.layers_by_name = {}
        self.build_properties = {}
        self.layer_templates = {}

    def build(self) -> Callable:
        def _create_model(input_shape, override_properties=None, output_size=None):
            model_properties = self.model_config.get("properties")
            if model_properties is not None:
                self.build_properties = {
                    **self.build_properties,
                    **model_properties
                }

            if override_properties is not None:
                self.build_properties = {
                    **self.build_properties,
                    **override_properties
                }

            if output_size is not None:
                self.build_properties = {
                    **self.build_properties,
                    'outputSize': output_size
                }

            input_config = self.model_config.get("input")
            num_inputs = 1
            input_join_type = None
            input_names = None
            if input_config is not None:
                if "num" in input_config:
                    num_inputs = input_config.get("num")
                if "join" in input_config:
                    input_join_type = input_config.get("join")

                input_names = input_config.get("names")

            input_layers = []
            for i in range(num_inputs):
                input_layer = self.builder_config.get_model_initializer()(
                    input_shape[i] if num_inputs > 1 else input_shape,
                    self.build_properties)
                input_layers.append(input_layer)
                self._add_layer(input_layer)

                self.layers_by_name["input" + str(i)] = input_layer
                if input_names is not None and len(input_names) > i:
                    self.layers_by_name[input_names[i]] = input_layer

            if input_join_type is not None:
                layer_builder = self.builder_registry.get_layer_builder(input_join_type)
                input_join_layer = layer_builder([], input_layers)
                self._add_layer(input_join_layer)
                self.layers_by_name["inputJoin"] = input_join_layer

            layer_templates_arr = self.model_config["templates"] if "templates" in self.model_config else []
            self.layer_templates = {item["name"]: item for item in layer_templates_arr}

            self._create_layers(self.model_config["layers"])

            model_creator = self.builder_registry.builder_config.get_model_creator()
            return model_creator(
                inputs=input_layers,
                layers=self.layers,
                layers_by_name=self.layers_by_name,
                output_config=self.model_config["output"] if "output" in self.model_config else {})

        return _create_model

    def _add_layer(self, layer, name=None):
        self.layers.append(layer)

        if "previous" in self.layers_by_name:
            self.layers_by_name["previous1"] = self.layers_by_name["previous"]
        self.layers_by_name["previous"] = layer
        self.layers_by_name["layer" + str(len(self.layers))] = layer
        if name is not None:
            self.layers_by_name[name] = layer

    def _create_layers(self, layers_config):
        for index, layer_config in enumerate(layers_config):
            if "--ignore" not in layer_config:
                layer = self._create_layer(layer_config)
                self._add_layer(layer, layer_config.get("name"))

    def _create_layer(self, layer_config):
        layer_connection = self._get_layer_connection(layer_config)

        if "template" in layer_config:
            layer_config = {
                **self.layer_templates[layer_config["template"]],
                **layer_config
            }

        def _layer_options_from_properties(name, default_value=None, repeat=0):
            if name in layer_config:
                value = layer_config[name]
            else:
                value = default_value

            def _find_value(_name, _default_value):
                if _name in self.build_properties:
                    return self.build_properties[_name]
                else:
                    return _default_value

            def _get_value(_name, _default_value):
                return self.builder_registry.expression_evaluator(
                    _default_value,
                    lambda _name: _find_value(_name, _default_value))

            if isinstance(value, (list,)):
                value = [_get_value(x, x) for x in value]
            else:
                value = _get_value(value, value)

            if repeat > 1:
                value = [value] * repeat

            return value

        def _layer_options(name, default_value=None, repeat=0):
            if self.builder_registry.has_layer_options_builder(name):
                return self.builder_registry.get_layer_options_builder(name)(
                    _layer_options_from_properties,
                    default_value)
            else:
                return _layer_options_from_properties(name, default_value, repeat)

        layer_type = _layer_options("type")
        layer_builder = self.builder_registry.get_layer_builder(layer_type)

        return layer_builder(_layer_options, layer_connection)

    def _get_layer_connection(self, layer_config):
        connection_to = "previous" \
            if "connectionTo" not in layer_config \
            else layer_config["connectionTo"]

        if isinstance(connection_to, list):
            layer_connection = [self.layers_by_name[con] for con in connection_to]
        else:
            layer_connection = self.layers_by_name[connection_to]

        return layer_connection

