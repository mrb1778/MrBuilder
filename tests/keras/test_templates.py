import unittest

from base.test_templates import TestBuilderTemplates
from test_bootstrap import TestBootstrapKeras


class TestBuilderTemplatesKeras(TestBuilderTemplates.Base, TestBootstrapKeras):
    pass


if __name__ == '__main__':
    unittest.main()
