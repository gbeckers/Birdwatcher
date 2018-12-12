from unittest import TestLoader, TextTestRunner, TestSuite

from . import test_videoinput
from . import test_movementdetection

modules = [test_videoinput, test_movementdetection]

def test(verbosity=1):
    suite =TestSuite()
    for module in modules:
        suite.addTests(TestLoader().loadTestsFromModule(module))
    return TextTestRunner(verbosity=verbosity).run(suite)