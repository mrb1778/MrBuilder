import unittest

from base.test_templates import TestBuilderTemplates
from test_bootstrap import TestBootstrapPytorch


class TestBuilderTemplatesPytorch(TestBuilderTemplates.Base, TestBootstrapPytorch.Base):
    pass


if __name__ == '__main__':
    unittest.main()
