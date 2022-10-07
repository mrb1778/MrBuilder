import unittest

from base.test_builder_control import TestBuilderControl
from test_bootstrap import TestBootstrapKeras


class TestBuilderControlKeras(TestBuilderControl.Base, TestBootstrapKeras):
    pass


if __name__ == '__main__':
    unittest.main()
