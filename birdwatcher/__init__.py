from .video import *
from .movementdetection import *
from .backgroundsubtraction import *
from .coordinatearrays import *
from .frameprocessing import *
from .utils import *
#from .plotting import *
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


from .tests import test