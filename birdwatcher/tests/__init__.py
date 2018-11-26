from unittest import TestLoader, TextTestRunner, TestSuite

from . import test_videoinput

modules = [test_videoinput]

def test(verbosity=1):
    suite =TestSuite()
    for module in modules:
        suite.addTests(TestLoader().loadTestsFromModule(module))
    return TextTestRunner(verbosity=verbosity).run(suite)