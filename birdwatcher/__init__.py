from .videoinput import *
from .movementdetection import *
from .bgsegm import *
from .coordinatearrays import *
from . import frameprocessing
#from .plotting import *
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


from .tests import test