"""This module provides objects and functions for coordinate data.
Coordinates are pixel positions (x,y) in a Frame. Coordinate Arrays are a
convenient way of storing output from image algorithms that determine if a
pixel is categorized as something (e.g. crossing some threshold).
Information is memory-mapped from disk, because data can easily become very
large and will not fit in RAM memory. Since the number of pixels per frame may
be variable, we use Ragged Arrays from the python library Darr to store them.
This is not the most disk-space efficient way of doing this (no compression),
but it is fast and the data can easily be read in any scientific computing
environement (Python, Matlab, R, Mathematica, etc.) Coordinate files can be
archived in compressed form (lzma) to save disk space.

"""

import os
import shutil
import tarfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from darr import RaggedArray, delete_raggedarray, create_raggedarray

from ._version import get_versions
from .utils import tempdir
from .frames import frameiterator


__all__ = ['CoordinateArrays', 'create_coordarray', 
           'open_archivedcoordinatedata', 'extract_archivedcoordinatedata', 
           'delete_coordinatearray', 'move_coordinatearrays']


def _archive(rar):
    rar.archive(overwrite=True)
    delete_raggedarray(rar)


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
    """A disk-based data type to store frame coordinates of consecutive
    frames.

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
            The sequence number of the frame to get.
        nchannels : int, optional
            The number of color channels in the frame. Default None leads
            to no color dimension, just a 2D frame with gray values.
        dtype : numpy dtype, default='uint8'
            Dtype of the returned frame.
        value : int, default=1
            The value to set the present coordinates with.

        Returns
        -------
        Numpy array

        """
        return _coordstoframe(coords=self[frameno], width=self.framewidth,
                              height=self.frameheight, nchannels=nchannels,
                              dtype=dtype, value=value)

    @frameiterator
    def iter_frames(self, startframe=0, endframe=None, stepsize=1,
                    nchannels=None, dtype='uint8', value=1):
        """Iterate over coordinate array and produce frames.

        Parameters
        ----------
        startframe : int, optional
            If specified, start iteration at this frame number.
        endfrom : int, optional
            Frame number to end iteration at. Default is None, which is to
            the end.
        stepsize : int, optional
            Step sizes. Defaults to 1, but if you want to skip frames, you
            can use this parameter.
        nchannels : int, optional
            The number of color channels in the frame. Default is None which
            leads to no color dimension, just a 2D frame with gray values.
        dtype : numpy dtype, default='uint8'
            Dtype of the returned frame.
        value : int, default=1
            The value to set the present coordinates with.

        Yields
        ------
        Frames
            Iterator that produces video frames based on the coordinates.

        """
        for coords in self.iter_arrays(startindex=startframe,
                                       endindex=endframe, stepsize=stepsize):
            yield _coordstoframe(coords=coords, width=self.framewidth,
                                 height=self.frameheight, nchannels=nchannels,
                                 dtype=dtype, value=value)

    def tovideo(self, filepath, startframe=0, endframe=None, stepsize=1,
                framerate=None, crf=17, scale=None, format='mp4',
                codec='libx264', pixfmt='yuv420p', ffmpegpath='ffmpeg'):
        """Writes frames based on coordinate info to a video file.

        Parameters
        ----------
        filepath : str
            Name of the videofilepath that should be written to.
        startframe: int, optional
            If specified, start iteration at this frame number.
        endfrom : int, optional
            Frame number to end iteration at. Default is None, which is to
            the end.
        stepsize : int, optional
            Step sizes. Defaults to 1, but if you want to skip frames, you
            can use this parameter.
        framerate : int, optional
            framerate of video in frames per second. If None, will look for
            `avgframerate` in metadata.
        crf : int, default=17
            Value determines quality of video. The default 17 is high quality.
            Use 23 for good quality.
        scale : tuple, optional
            (width, height). The default (None) does not change width and
            height.
        format : str, default='mp4'
            ffmpeg video format.
        codec : str, default='libx264'
            ffmpeg video codec.
        pixfmt : str, default='yuv420p'
            ffmpeg pixel format.
        ffmpegpath : str or pathlib.Path, optional
            Path to ffmpeg executable. Default is `ffmpeg`, which means it
            should be in the system path.

        Notes
        -----
        See ffmpeg documentation for more information.

        """
        from .ffmpeg import arraytovideo
        if framerate is None:
            try:
                framerate = self.metadata['avgframerate']
            except KeyError:
                raise ValueError('Cannot find a frame rate, you need to '
                                 'provide one with the `framerate` parameter')
        arraytovideo(self.iter_frames(startframe=startframe,
                                      endframe=endframe,
                                      stepsize=stepsize, nchannels=3,
                                      value=255,
                                      dtype='uint8'),
                     filepath, framerate=framerate, scale=scale, crf=crf,
                     format=format, codec=codec, pixfmt=pixfmt,
                     ffmpegpath=ffmpegpath)

    def show(self, startframe=0, endframe=None, stepsize=1, framerate=None,
             draw_framenumbers=True):
        """Shows coordinates frames in a video window.

        Turns each coordinate array into a frame and then plays video.

        Parameters
        ----------
        startframe: int, optional
            If specified, start iteration at this frame number.
        endfrom : int, optional
            Frame number to end iteration at. Default is None, which is to
            the end.
        stepsize : int, optional
            Step sizes. Defaults to 1, but if you want to skip frames, you
            can use this parameter.
        framerate : int, optional
            framerate of video in frames per second. If None, will look for
            `avgframerate` in metadata.
        draw_framenumbers : bool, default=True
            Should I draw frame numbers yes or no?

        Returns
        -------

        """
        if framerate is None:
            try:
                framerate = self.metadata['avgframerate']
            except KeyError:
                raise ValueError('Cannot find a frame rate, you need to '
                                 'provide one with the `framerate` parameter')

        f = self.iter_frames(startframe=startframe, endframe=endframe,
                         stepsize=stepsize, nchannels=3, value=255,
                         dtype='uint8')
        if draw_framenumbers:
            f = f.draw_framenumbers()
        return f.show(framerate=framerate)

    def get_coordcount(self, startframeno=0, endframeno=None):
        """Get the number of coordinates present per frame.

        Parameters
        ----------
        startframeno : int, optional
            Defaults to the beginning of the coordinate array.
        endframeno : int, optional
            Defaults to the end of the coordinate array.

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
        startframeno : int, optional
            Defaults to the beginning of the coordinate array.
        endframeno : int, optional
            Defaults to the end of the coordinate array.

        Returns
        -------
        Numpy Array
            Sequence of numbers, each with a coordinate mean.

        """
        coordgen = self.iter_arrays(startindex=startframeno,
                                    endindex=endframeno)
        return np.array([c.mean(0) if c.size>0 else (np.nan, np.nan)
                         for c in coordgen])

    def get_coordmedian(self, startframeno=0, endframeno=None):
        """Get the median of the coordinates per frame.

        Parameters
        ----------
        startframeno : int, optional
            Defaults to the beginning of the coordinate array.
        endframeno : int, optional
            Defaults to the end of the coordinate array.

        Returns
        -------
        Numpy Array
            Sequence of numbers, each with a coordinate median.

        """
        coordgen = self.iter_arrays(startindex=startframeno,
                                    endindex=endframeno)
        return np.array([np.median(c,0) if c.size>0 else (np.nan, np.nan)
                         for c in coordgen])


def create_coordarray(path, framewidth, frameheight, metadata=None,
                      overwrite=True):
    """Creates an empty Coordinate Arrays object.

    Parameters
    ----------
    path : str
        Path to disk-based coordinate array directory that should be written
        to.
    framewidth : int
        Width in pixels.
    frameheight : int
        Height in pixels.
    metadata : dict, optional
    overwrite : bool, default=True
        Overwrite existing CoordinateArrays or not.

    Returns
    -------
    CoordinateArrays

    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)   
    if metadata is None:
        metadata = {}
    metadata.update({'framewidth': framewidth,
                     'frameheight': frameheight,
                     'birdwatcher_version': get_versions()['version']})
    coords = create_raggedarray(path, atom=(2,), metadata=metadata,
                                overwrite=overwrite, dtype='uint16')
    return CoordinateArrays(coords.path, accessmode='r+')


@ contextmanager
def open_archivedcoordinatedata(path, temppath=None):
    """A context manager that temporarily decompresses coordinate
    data to work with coordinate array.

    Parameters
    ----------
    path : str
        Path to the archive.

    Yields
    ------
    Context manager to work with temporarily uncompressed coordinate
    array.

    Examples
    --------
    >>> with open_archivedcoordinatedata('coords.darr.tar.xz') as coords:
            # do stuff with coordinate array

    """
    path = Path(path)
    if not path.suffix == '.xz':
        raise OSError(f'{path} does not seem to be archived coordinate data')

    with tempdir(dirname=temppath) as dirname:
        tar = tarfile.open(path)
        tar.extractall(path=dirname)
        tar.close()
        capath = list(dirname.glob('*'))[0]
        yield CoordinateArrays(capath)


def extract_archivedcoordinatedata(path):
    """An archived coordinate array (.tar.xz) is decompressed.

    Parameters
    ----------
    path : str
        Path to the archived coordinate array.

    Returns
    -------
    CoordinateArrays

    """
    path = Path(path)
    if not path.suffix == '.xz':
        raise OSError(f'{path} does not seem to be archived coordinate data')

    tar = tarfile.open(path)
    tar.extractall(path=path.parent)
    tar.close()
    
    while path.suffix in {'.tar', '.xz'}: # remove extensions
        path = path.with_suffix('')

    return CoordinateArrays(path)


delete_coordinatearray = delete_raggedarray


def move_coordinatearrays(sourcedirpath, targetdirpath):
    """Move coordinate / darr data hierarchically out of a source dir and
    move it to a target dir, keeping the hierarchy intact.

    This is handy when you created a zillion coordinate / darr inside some
    directory hierarchy of input data, and you want to separate things.

    Parameters
    ----------
    sourcedirpath : str or Path
    targetdirpath : str or Path
        The top-level directory to which everything is moved. If it doesn't
        exist it will be created.

    """
    tdir = Path(targetdirpath)
    for root, dirs, files in os.walk(sourcedirpath):
        for dname in dirs:
            if dname.endswith('.darr'):
                d = Path(dname)
                newdir = tdir / Path(root).relative_to(sourcedirpath)
                Path(newdir).mkdir(parents=True, exist_ok=True)
                shutil.move(f"{root}/{d}", str(newdir))