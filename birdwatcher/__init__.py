from .video import *
from .movementdetection import *
from .backgroundsubtraction import *
from .coordinatearrays import *
from .frames import *
from .parameters import *
from .utils import *
try:
    import pypylon
    from .recording import *
except ImportError:
    pass

#from .plotting import *
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


from .tests import test