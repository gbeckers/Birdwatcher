from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("birdwatcher")
except PackageNotFoundError:
    __version__ = "unknown"

from .video import *
from .backgroundsubtraction import *
from .coordinatearrays import *
from .frames import *
from .ffmpeg import supported_audio_codecs, supported_video_codecs
