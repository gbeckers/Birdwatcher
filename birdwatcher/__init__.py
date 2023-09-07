from .video import *
from .backgroundsubtraction import *
from .coordinatearrays import *
from .frames import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .tests import test