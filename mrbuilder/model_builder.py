from typing import Callable, Dict

from mrbuilder.variable_registry import VariableRegistry


class ModelBuilderInstance:
    def __init__(self, builder_registry, model_config, variable_registry) -> None:
        super().__init__()
        self.builder_registry = builder_registry
        self.model_config = model_config

        self.variable_registry = variable_registry
        self.layers = []
        self.layers_by_name = {}
        self.layer_templates = {}

    def create_inputs(self, input_shape):
        input_config = self.model_config.get("inputs")
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
        model_input_builder = self.builder_registry.get_model_input_builder()
        for i in range(num_inputs):
            input_layer = model_input_builder(
                input_shape[i] if num_inputs > 1 else input_shape,
                self.variable_registry.find)
            input_layers.append(input_layer)
            self._add_layer(input_layer)

            self.layers_by_name["input" + str(i)] = input_layer
            if input_names is not None and len(input_names) > i:
                self.layers_by_name[input_names[i]] = input_layer
        if input_join_type is not None:
            layer_builder = self.builder_registry.get_layer_builder(input_join_type)
            input_join_layer = layer_builder(self._get_variable_from_registry, input_layers)
            self._add_layer(input_join_layer)
            self.layers_by_name["inputJoin"] = input_join_layer
        return input_layers

    def _add_layer(self, layer, name=None):
        self.layers.append(layer)

        if "previous" in self.layers_by_name:
            self.layers_by_name["previous1"] = self.layers_by_name["previous"]
        self.layers_by_name["previous"] = layer
        self.layers_by_name["layer" + str(len(self.layers))] = layer
        if name is not None:
            self.layers_by_name[name] = layer

    def create_layers(self, layers_config, layers_options=None):
        for index, layer_config in enumerate(layers_config):
            context_depth_initial = self.variable_registry.get_context_depth()

            self.variable_registry.set_scoped()
            self.variable_registry.push_context(layers_options)

            if "template" in layer_config:
                self.variable_registry.push_context(self.layer_templates[layer_config["template"]])
            self.variable_registry.push_context(layer_config)
            self.variable_registry.push_value("template", False)

            if self.variable_registry.find("if") is None or self.variable_registry.find("if"):
                layers = self.variable_registry.find("layers")
                if layers:
                    self.variable_registry.push_value("layers", False)
                    self.create_layers(layers, layer_config)
                else:
                    layer = self._create_layer(layer_config)
                    self._add_layer(layer, self.variable_registry.find("name"))

            self.variable_registry.pop_context_to_depth(context_depth_initial)
            self.variable_registry.set_scoped(False)

    def _create_layer(self, layer_config):
        layer_connection = self._get_layer_connection(layer_config)

        layer_type = self._get_variable_from_registry("type")
        layer_builder = self.builder_registry.get_layer_builder(layer_type)
        layer = layer_builder(self._get_variable_from_registry, layer_connection)
        return layer

    def _get_variable_from_registry(self, name, default_value=None, repeat=0):
        if self.builder_registry.has_layer_attribute_builder(name):
            return self.builder_registry.get_layer_attribute_builder(name)(
                self._get_raw_variable_from_registry,
                default_value)
        else:
            return self._get_raw_variable_from_registry(name, default_value, repeat)

    def _get_raw_variable_from_registry(self, name, default_value=None, repeat=0):
        value = self.variable_registry.find(name, default_value)

        if isinstance(value, (list,)):
            value = [self.variable_registry.find(x, x) for x in value]
        else:  # todo: dont recurse unless is macro --> unwrap macros to macros
            value = self.variable_registry.find(value, value)

        if repeat > 1:
            value = [value] * repeat

        return value

    def _get_layer_connection(self, layer_config):
        connection_to = "previous" \
            if "connectionTo" not in layer_config \
            else layer_config["connectionTo"]

        if isinstance(connection_to, list):
            layer_connection = [self.layers_by_name[con] for con in connection_to]
        else:
            layer_connection = self.layers_by_name[connection_to]

        return layer_connection


class ModelBuilder:
    # builder_registry: BuilderRegistry

    layers: list
    layers_by_name: Dict
    properties: Dict
    variable_registry: VariableRegistry

    def __init__(self, builder_registry, model_config: Dict, name=None) -> None:
        super().__init__()
        self.builder_registry = builder_registry
        self.model_config = model_config

        self.model_properties = self.model_config.get("properties")
        self.name = name

    def build(self) -> Callable:
        def _create_model(input_shape, model_args=None, **kwargs):
            variable_registry = VariableRegistry(self.builder_registry.expression_evaluator,
                                                 self.model_properties)

            model_args_merged = {**model_args, **kwargs} if model_args is not None else model_args
            if model_args is not None and kwargs is not None:
                # todo: snake case to camel
                variable_registry.push_context(model_args_merged)
                if "output_size" in model_args_merged:
                    variable_registry.push_value('outputSize', model_args_merged["output_size"])

            # todo: move to 1 call
            model_builder_instance = ModelBuilderInstance(self.builder_registry,
                                                          self.model_config,
                                                          variable_registry)

            input_layers = model_builder_instance.create_inputs(input_shape)

            layer_templates_arr = model_builder_instance.model_config["templates"] if "templates" in self.model_config else []
            model_builder_instance.layer_templates = {item["name"]: item for item in layer_templates_arr}

            model_builder_instance.create_layers(model_builder_instance.model_config["layers"])

            model_creator = self.builder_registry.get_model_creator()

            return model_creator(
                inputs=input_layers,
                layers=model_builder_instance.layers,
                layers_by_name=model_builder_instance.layers_by_name,
                output_config=self.model_config["output"] if "output" in self.model_config else {},
                name=self.name)

        return _create_model


class MissingLayerTypeException(Exception):
    pass
