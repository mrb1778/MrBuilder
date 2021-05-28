import unittest

from base.test_builder import TestBuilder
from test_bootstrap import TestBootstrapPytorch


class TestBuilderPytorch(TestBuilder.Base, TestBootstrapPytorch.Base):
    pass


if __name__ == '__main__':
    unittest.main()
