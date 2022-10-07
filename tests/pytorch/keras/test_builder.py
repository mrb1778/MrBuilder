import unittest

from base.test_builder import TestBuilder
from test_bootstrap import TestBootstrapKeras


class TestBuilderKeras(TestBuilder.Base, TestBootstrapKeras):
    pass


if __name__ == '__main__':
    unittest.main()
