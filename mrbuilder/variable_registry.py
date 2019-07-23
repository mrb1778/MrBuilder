from collections import deque
from typing import Dict, Any

# from mrbuilder.builder_registry import BuilderRegistry


class VariableRegistry:
    contexts: deque
    scope_index: int
    # builder_registry: BuilderRegistry

    def __init__(self, builder_registry, initial_context: Dict = None) -> None:
        super().__init__()

        self.builder_registry = builder_registry

        self.contexts = deque()
        self.scope_index = -1
        self.push_context(initial_context)

    def set_scoped(self, scoped=True) -> None:
        self.scope_index = len(self.contexts) if scoped else -1

    def push_context(self, context: Dict) -> None:
        if context is not None:
            self.contexts.appendleft(context)

    def push_value(self, name: str, value: Any) -> None:
        if name is not None and value is not None:
            context = {name: value}
            self.push_context(context)

    def pop_context(self) -> Dict:
        return self.contexts.popleft()

    def get_context_depth(self) -> int:
        return len(self.contexts)

    def pop_context_to_depth(self, depth: int):
        while self.get_context_depth() < depth:
            self.pop_context()

    def find(self, name, default_value=None, limit_to_scope=True):
        value = self._find_in_context(name, limit_to_scope)
        if value is None:
            value = default_value

        if self.builder_registry.expression_evaluator.is_expression(value):
            return self.builder_registry.expression_evaluator.eval(value, self.get)
        else:
            return value

    def _find_in_context(self, name, limit_to_scope=True):
        for i, context in enumerate(self.contexts):
            num_contexts = len(self.contexts)
            if limit_to_scope \
                    and self.scope_index != -1 \
                    and num_contexts - i < self.scope_index:
                return None
            if name in context:
                return context[name]
        return None

    def get(self, name):
        value = self.find(name, limit_to_scope=False)
        if value is None:
            raise VariableNotFoundException("Not Found{}".format(name))
        return value


class VariableNotFoundException(Exception):
    pass
