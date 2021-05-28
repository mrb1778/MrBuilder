import unittest

from base.test_builder_control import TestBuilderControl
from test_bootstrap import TestBootstrapPytorch


class TestBuilderControlPytorch(TestBuilderControl.Base, TestBootstrapPytorch.Base):
    pass


if __name__ == '__main__':
    unittest.main()
