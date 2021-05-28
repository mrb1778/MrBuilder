import unittest

from base.test_registry import TestBuilderRegistry
from test_bootstrap import TestBootstrapPytorch


class BuilderRegistryTestPytorch(TestBuilderRegistry.Base, TestBootstrapPytorch.Base):
    pass


if __name__ == '__main__':
    unittest.main()
