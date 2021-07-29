class ExpressionEvaluator:
    def is_expression(self, expression) -> bool:
        raise NotImplementedError

    def eval(self, expression, variable_resolver):
        raise NotImplementedError
