import unittest

from base.test_model_library import TestModelLibrary
from test_bootstrap import TestBootstrapKeras


class TestModelLibraryKeras(TestModelLibrary.Base, TestBootstrapKeras):
    pass


if __name__ == '__main__':
    unittest.main()
