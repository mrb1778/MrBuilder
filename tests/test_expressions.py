import unittest

from mrbuilder.expressions.sexpression import SimpleExpressionEvaluator


class ExpressionsTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.expression_evaluator = SimpleExpressionEvaluator() 
        self.variables = {
            "var1": 1,
            "var2": 2
        }
        # todo: expose variable resolver for testing
        self.variable_resolver = lambda name: self.variables[name] if name in self.variables else name

    def test_non_expression(self):
        # self.assertEqual(self.expression_evaluator.eval("1", self.variable_resolver), 1)
        # self.assertEqual(self.expression_evaluator.eval("2.5", self.variable_resolver), 2.5)
        self.assertEqual(self.expression_evaluator.eval("bob", self.variable_resolver), "bob")

    def test_constant_single_expression(self):
        self.assertEqual(self.expression_evaluator.eval("{{2.5}}", self.variable_resolver), 2.5)

    def test_variable_single_expression(self):
        self.assertEqual(self.expression_evaluator.eval("{{var1}}", self.variable_resolver), self.variables["var1"])

    def test_constant_constant_addition(self):
        self.assertEqual(self.expression_evaluator.eval("{{2 + 3}}", self.variable_resolver), 2 + 3)

    def test_variable_constant_addition(self):
        self.assertEqual(self.expression_evaluator.eval("{{var1 + 3}}", self.variable_resolver),
                         self.variables["var1"] + 3)

    def test_constant_variable_addition(self):
        self.assertEqual(self.expression_evaluator.eval("{{3 + var1}}", self.variable_resolver),
                         3 + self.variables["var1"])

    def test_variable_variable_addition(self):
        self.assertEqual(self.expression_evaluator.eval("{{var1 + var2}}", self.variable_resolver),
                         self.variables["var1"] + self.variables["var2"])

    def test_variable_variable_multiplication(self):
        self.assertEqual(self.expression_evaluator.eval("{{var1 * var2}}", self.variable_resolver),
                         self.variables["var1"] * self.variables["var2"])

    def test_variable_variable_or(self):
        self.assertEqual(self.expression_evaluator.eval("{{var1 || var2}}", self.variable_resolver),
                         self.variables["var1"])
        self.assertEqual(self.expression_evaluator.eval("{{0 || var2}}", self.variable_resolver),
                         self.variables["var2"])


if __name__ == '__main__':
    unittest.main()
