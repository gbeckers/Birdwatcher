from unittest import TestLoader, TextTestRunner, TestSuite

from . import test_coordinatearrays
from . import test_frameprocessing
from . import test_movementdetection
from . import test_plotting
from . import test_videoinput
from . import test_backgroundsubtraction


modules = [test_coordinatearrays, test_frameprocessing,
           test_movementdetection, test_plotting, test_videoinput, 
           test_backgroundsubtraction]

def test(verbosity=1):
    suite =TestSuite()
    for module in modules:
        suite.addTests(TestLoader().loadTestsFromModule(module))
    return TextTestRunner(verbosity=verbosity).run(suite)