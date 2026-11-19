"""Version information of Birdwatcher and the software it depends on.

This is used to store provenance information as metadata with analysis
results.

"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

import cv2 as cv
import numpy as np

from . import __version__
from .ffmpeg import ffmpegversion


__all__ = ["versions"]


# OpenCV can be installed from different pip distributions that provide the
# same `cv2` module, but not the same functionality (e.g., `cv2.bgsegm` is
# only available in the contrib builds).
_opencv_distributions = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)


def opencv_distribution() -> str | None:
    """Name of the installed pip distribution that provides OpenCV.

    Returns
    -------
    str or None
        Distribution name and version, e.g. 'opencv-contrib-python 5.1.0.95',
        or None if OpenCV was not installed via pip (e.g., a system or conda
        build).

    """
    found = []
    for name in _opencv_distributions:
        try:
            found.append(f"{name} {_pkg_version(name)}")
        except PackageNotFoundError:
            pass
    return ", ".join(found) if found else None


def opencvversions() -> dict[str, str | None]:
    """Version information on OpenCV.

    Returns
    -------
    dict
        Dictionary with the OpenCV version and the pip distribution that
        provides it.

    """
    return {
        "opencv": cv.__version__,
        "opencvdistribution": opencv_distribution(),
    }


def versions() -> dict[str, str | None]:
    """Versions of Birdwatcher and the important software it depends on.

    Returns
    -------
    dict
        Dictionary with version strings of birdwatcher, ffmpeg, opencv,
        numpy and darr.

    """
    return {
        "birdwatcher": __version__,
        "ffmpeg": ffmpegversion(),
        **opencvversions(),
        "numpy": np.__version__,
        "darr": _pkg_version("darr"),
    }
