from collections import deque
from typing import Dict, Any

from mrbuilder.expressions.expression_evaluator import ExpressionEvaluator


class VariableRegistry:
    contexts: deque
    scope_index: int
    expression_evaluator: ExpressionEvaluator

    def __init__(self, expression_evaluator, initial_context: Dict = None) -> None:
        super().__init__()

        self.expression_evaluator = expression_evaluator

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
        while self.get_context_depth() > depth:
            self.pop_context()

    def find(self, name, default_value=None, limit_to_scope=True, start_at_depth=0):
        value, context_depth = self._find_in_context(name, limit_to_scope, start_at_depth)
        if value is None:
            value = default_value

        if self.expression_evaluator.is_expression(value):
            return self.expression_evaluator.eval(
                value,
                lambda get_name: self.get(get_name, context_depth + 1))
        else:
            return value

    def _find_in_context(self, name, limit_to_scope=True, start_at_depth=0):
        for (i, context) in list(enumerate(self.contexts))[start_at_depth:]:
            num_contexts = len(self.contexts)
            if limit_to_scope \
                    and self.scope_index != -1 \
                    and num_contexts - i < self.scope_index:
                return None, i
            if name in context:
                return context[name], i
        return None, None

    def get(self, name, start_at_depth=0):
        value = self.find(name, limit_to_scope=False, start_at_depth=start_at_depth)
        if value is None:
            raise VariableNotFoundException("Not Found{}".format(name))
        return value


class VariableNotFoundException(Exception):
    pass
