import unittest

from base.test_registry import TestBuilderRegistry
from test_bootstrap import TestBootstrapKeras


class BuilderRegistryTestKeras(TestBuilderRegistry.Base, TestBootstrapKeras):
    pass


if __name__ == '__main__':
    unittest.main()
