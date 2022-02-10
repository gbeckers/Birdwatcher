"""This module provides objects and functions for coordinate data.
Coordinates are pixel positions (x,y). This is a convenient way of storing
output from image algorithms that determine whether or not a pixel is
categorized as something (e.g. crossing some threshold). Since the number of
pixels per frame may be different, we use disk-based ragged arrays from the
python library Darr to store them. This is not the most disk-space efficient
way of doing this, but it is fast and the data can easily be read in any
computing platform. Coordinate files can be archived in compressed form
(lzma) when data is large.

"""

import os
from contextlib import contextmanager
from pathlib import Path
import shutil
import numpy as np
import tarfile

from darr import RaggedArray, delete_raggedarray, create_raggedarray

from ._version import get_versions
from .utils import tempdir
from .frameprocessing import frameiterator

__all__ = ['CoordinateArrays', 'open_archivedcoordinatedata',
           'create_coordarray']


def _coordstoframe(coords, width, height, nchannels=None, dtype='uint8',
                   value=1):
    if nchannels is None:
        frame = np.zeros((height, width), dtype=dtype)
    else:
        frame = np.zeros((height, width, nchannels), dtype=dtype)
    frame[(coords[:, 1], coords[:, 0])] = value
    return frame


# fixme, should we allow for 3 values: (x, y, val)?
class CoordinateArrays(RaggedArray):
    """A disk-based data type to store frame coordinates of consecutive frames.

    Maximum for frame width and height is 65535.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to disk-based coordinate array directory.
    accessmode : {'r', 'r+'}, default 'r'
       File access mode of the darr data. `r` means read-only, `r+` means
       read-write. `w` does not exist. To create new coordinate arrays,
       potentially overwriting an other one, use the `create_coordarray`
       functions.


    """

    def __init__(self, path, accessmode='r'):
        super().__init__(path=path, accessmode=accessmode)
        md = dict(self.metadata)
        self.framewidth = md['framewidth']
        self.frameheight = md['frameheight']

    def get_frame(self, frameno, nchannels=None, dtype='uint8', value=1):
        """Get a frame based on a sequence number in the coordinate array.

        Parameters
        ----------
        frameno : int
            The sequence number of the frame to get
        nchannels : int
            The number of color channels in the frame. Default is None which
            leads to no color dimension, just a 2D frame with gray values.
        dtype :
            Numpy dtype of the returned frame. Defaults to unit8
        value:
            The value to set the present coordinates with. Default is 1.

        Returns
        -------
        Numpy array


        """

        return _coordstoframe(coords=self[frameno], width=self.framewidth,
                              height=self.frameheight, nchannels=nchannels,
                              dtype=dtype, value=value)

    @frameiterator
    def iter_frames(self, startframe=0, endframe=None, stepsize=1, nchannels=None,
                    dtype='uint8', value=1):
        """Iterate over coordinate array and produce frames.

        Parameters
        ----------
        startframe: int
            Frame number to start iteration at. Default is 0.
        endfrom: int or None
            Frame number to end iteration at. Default is None, which is to
            the end.
        stepsize: int
            Step sizes. Defaults to 1, but if you want to skip frames, you
            can use this parameter.
        nchannels : int
            The number of color channels in the frame. Default is None which
            leads to no color dimension, just a 2D frame with gray values.
        dtype :
            Numpy dtype of the returned frame. Defaults to unit8
        value:
            The value to set the present coordinates with. Default is 1.

        Returns
        -------
        Frames
            Iterator that produces video frames based on the coordinates.

        """
        for coords in self.iter_arrays(startindex=startframe, endindex=endframe, stepsize=stepsize):
            yield _coordstoframe(coords=coords, width=self.framewidth,
                                 height=self.frameheight, nchannels=nchannels,
                                 dtype=dtype, value=value)

    def tovideo(self, filepath, startframe=0, endframe=None, stepsize=1,
                framerate=None, scale=None, crf=17, format='mp4',
                codec='libx264', pixfmt='yuv420p', ffmpegpath='ffmpeg'):
        """Writes frames based on coordinate info to a video file.

        Parameters
        ----------
        filename: str
            Name of the videofile that should be written to
        startframe: int
            Frame number to start iteration at. Default is 0.
        endfrom: int or None
            Frame number to end iteration at. Default is None, which is to
            the end.
        stepsize: int
            Step sizes. Defaults to 1, but if you want to skip frames, you
            can use this parameter.
        framerate: int
            framerate of video in frames per second
        crf: int
            Value determines quality of video. Default: 23, which is good
            quality. See ffmpeg documentation. Use 17 for high quality.
        scale: tuple or None
            (width, height). If None, do not change width and height.
            Default: None.
        format: str
            ffmpeg video format. Default is 'mp4'. See ffmpeg documentation.
        codec: str
            ffmpeg video codec. Default is 'libx264'. See ffmpeg documentation.
        pixfmt: str
            ffmpeg pixel format. Default is 'yuv420p'. See ffmpeg documentation.
        ffmpegpath: str or pathlib.Path
            Path to ffmpeg executable. Default is `ffmpeg`, which means it
            should be in the system path.

        """
        from .ffmpeg import arraytovideo
        if framerate is None:
            try:
                framerate = self.metadata['avgframerate']
            except KeyError:
                raise ValueError('Cannot find a frame rate, you need to '
                                 'provide one with the `framerate` parameter')
        arraytovideo(self.iter_frames(startframe=startframe, endframe=endframe,
                                      stepsize=stepsize, nchannels=3, value=255,
                                      dtype='uint8'),
                     filepath, framerate=framerate, scale=scale, crf=crf,
                     format=format, codec=codec, pixfmt=pixfmt,
                     ffmpegpath=ffmpegpath)


    def get_coordcount(self, startframeno=0, endframeno=None):
        """Get the number of coordinates present per frame.

        Parameters
        ----------
        startframeno : int
            Default is 0.
        endframeno : int
            Defaults to None, which is the end of the coordinate array

        Returns
        -------
        Numpy Array
            Sequence of numbers, each with a coordinate count.

        """
        coordgen = self.iter_arrays(startindex=startframeno,
                                         endindex=endframeno)
        return np.array([c.shape[0] for c in coordgen])

    def get_coordmean(self, startframeno=0, endframeno=None):
        """Get the mean of the coordinates per frame.

        Parameters
        ----------
        startframeno : int
            Default is 0.
        endframeno : int
            Defaults to None, which is the end of the coordinate array

        Returns
        -------
        Numpy Array
            Sequence of numbers, each with a coordinate mean.

        """
        coordgen = self.iter_arrays(startindex=startframeno,
                                         endindex=endframeno)
        return np.array([c.mean(0) for c in coordgen])


def create_coordarray(path, framewidth, frameheight, metadata=None,
                      overwrite=True):
    """Creates an empty Coordinate Arrays object.

    Parameters
    ----------
    path
    framewidth
    frameheight
    metadata
    overwrite

    Returns
    -------
    CoordinateArrays

    """
    if metadata is None:
        metadata = {}
    metadata.update({'framewidth': framewidth,
                     'frameheight': frameheight,
                     'birdwatcher_version': get_versions()['version']})
    coords = create_raggedarray(path, atom=(2,), metadata=metadata,
                                overwrite=overwrite, dtype='uint16')
    return CoordinateArrays(coords.path, accessmode='r+')


delete_coordinatearray = delete_raggedarray


@ contextmanager
def open_archivedcoordinatedata(path):
    """A context manager that temporarily decompresses coordinate
    data to work with coordinate array.

    Parameters
    ----------
    path: str
        Path to the archive.

    Returns
    -------
    Context manager to work with temporarily uncompressed coordinate
    array.

    Examples
    --------
    >>> with open_archivedcoordinatedata('coord.tar.xz') as coords:
            # do stuff with coordinate array

    """
    path = Path(path)
    if not path.suffix == '.xz':
        raise OSError(f'{path} does not seem to be archived coordinate data')

    with tempdir() as dirname:
        tar = tarfile.open(path)
        tar.extractall(path=dirname)
        tar.close()
        p = path.parts[-1].split('.tar.xz')[0]
        yield CoordinateArrays(Path(dirname) / Path(p))


def move_coordinatearrays(sourcedirpath, targetdirpath):
    """Move coordinate / darr data hierarchically out of a source dir and
    move it to a target dir, keeping the hierarchy intact.

    The is handy when you created a zillion coordinate / darr inside some
    directory hierarchy of input data, and you want to separate things.

    Parameters
    ----------
    sourcedirpath: str or Path

    targetdirpath: str or Path
        The top-level directory to which everything is moved. If it doesn't
        exist it will be created.

    """
    tdir = Path(targetdirpath)
    for root, dirs, files in os.walk(sourcedirpath):
        for dname in dirs:
            if dname.endswith('.darr'):
                d = Path(dname)
                newdir = tdir / root
                print(d, newdir)
                os.makedirs(newdir, exist_ok=True)
                shutil.move(f"{root}/{d}", str(newdir))
