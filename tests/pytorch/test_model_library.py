import unittest

from base.test_model_library import TestModelLibrary
from test_bootstrap import TestBootstrapPytorch


class TestModelLibraryPytorch(TestModelLibrary.Base, TestBootstrapPytorch.Base):
    pass


if __name__ == '__main__':
    unittest.main()
