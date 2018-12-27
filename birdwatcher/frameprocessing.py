"""This module contains classes for generating image frames and general
processing functionality such as measurement, drawing of labels/text and
saving as video files.

A general tip is to look at OpenCV's documentation if you want to understand
the parameters.


"""

import numpy as np
import cv2 as cv
from functools import wraps

from .utils import peek_iterable

__all__ = ['Frames', 'FramesColor', 'FramesGray', 'FramesNoise', 'framecolor',
           'framegray', 'framenoise']

def frameiterator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return Frames(func(*args, **kwargs))
    return wrapper


class Frames:
    """An iterator of frames with useful methods.

    This is a main base class in Birdwatcher, as many functions and
    methods return this type or can use it as input. It also has useful methods
    for final output, such as a video file, measurement, or adding labels.

    Parameters
    ----------
    frames: iterable
        This can be anything that is iterable and that produces image frames. A
        numpy array, a VideoFile or another Frames object.

    Examples
    --------
     >>> import birdwatcher as bw
     >>> frames = bw.FramesNoise(250, height=720, width=1280)
     >>> frames = frames.draw_framenumbers()
     >>> frames.tovideo('noisewithframenumbers.mp4', framerate=25)

    """

    def __init__(self, frames):

        first, frames = peek_iterable(frames)

        framewidth, frameheight, *nchannels = first.shape
        if nchannels == []:
            nchannels = 1
        self._frames = frames
        self._frameheight = frameheight
        self._framewidth = framewidth
        self._nchannels = nchannels
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._frames)

    @property
    def frameheight(self):
        self._frameheight

    @property
    def framewidth(self):
        self._framewidth

    @property
    def nchannels(self):
        self._nchannels

    def tovideo(self, filename, framerate, crf=23, format='mp4',
                codec='libx264', pixfmt='yuv420p', ffmpegpath='ffmpeg'):
        """Writes frames to video file.

        Parameters
        ----------
        filename: str
            Name of the videofile that should be written to
        framerate: int
            framerate of video in frames per second
        crf: int
            Value determines quality of video. Default: 23, which is good
            quality. See ffmpeg documentation. Use 17 for high quality.
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
        arraytovideo(frames=self, filename=filename, framerate=framerate,
                     crf=crf, format=format, codec=codec, pixfmt=pixfmt,
                     ffmpegpath=ffmpegpath)

    @frameiterator
    def blur(self, ksize, anchor=(-1,-1), borderType=cv.BORDER_DEFAULT):
        """Blurs frames using the normalized box filter.

        Parameters
        ----------
        ksize: (int, int)
            Kernel size. Tuple of ints.
        anchor: (int, int)
            Anchor point. Default value (-1,-1) means that the anchor is at
            the kernel center.
        borderType: int
            Border mode used to extrapolate pixels outside of the image.
            Default: 4.

        Returns
        -------
        Frames
            Iterator that generates blurred frames.

        Examples
        --------
        >>> import birdwatcher as bw
        >>> frames = bw.FramesNoise(250, height=720, width=1280)
        >>> frames = frames.blur(ksize=(10,10))
        >>> frames.tovideo('noiseblurred.mp4', framerate=25)

        """
        for frame in self._frames:
            yield cv.blur(frame, ksize=ksize, anchor=anchor,
                          borderType=borderType)

    @frameiterator
    def draw_circles(self, centers, radius=6, color=(255, 100, 0),
                     thickness=2, linetype=cv.LINE_AA, shift=0):
        """Draws circles on frames.

        Centers should be an iterable that has a length that corresponds to
        the number of frames.

        Parameters
        ----------
        centers: iterable
            Iterable that generate center coordinates (x, y) of the circles
        radius: int
            Radius of circle. Default 4.
        color: tuple of ints
            Color of circle (r, g, b). Default (255, 100, 0)
        thickness: int
            Line thickness. Default 2.
        linetype: int
            OpenCV line type of circle boundary. Default cv2.LINE_AA
        shift: int
            Number of fractional bits in the coordinates of the center and in
            the radius value. Default 0.

        Returns
        -------
        Frames
            Iterator that generates frames with circles

        """

        for frame, center in zip(self._frames, centers):
            center = np.asanyarray(center)
            if not np.isnan(center).any():
                (x, y) = center.astype('int16')
                yield cv.circle(frame, center=(x, y), radius=radius,
                                color=color,
                                thickness=thickness, lineType=linetype,
                                shift=shift)
            else:
                yield frame

    #FIXME check if input is color
    @frameiterator
    def togray(self):
        """Converts color frames to gray frames

        Returns
        -------
        Frames

        """
        for frame in self._frames:
            yield cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # FIXME check if input is gray
    @frameiterator
    def tocolor(self):
        """Converts gray frames to color frames

        Returns
        -------
        Frames

        """
        for frame in self._frames:
            yield cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    @frameiterator
    def draw_framenumber(self, startat=0, org=(2, 25),
                         fontface=cv.FONT_HERSHEY_SIMPLEX,
                         fontscale=1, color=(200, 200, 200), thickness=2,
                         linetype=cv.LINE_AA):
        """Draws the frame number on frames.

        Parameters
        ----------
        startat: int
            The number to start counting at. Default 0
        org: 2-tuple of ints
            Bottom-left corner of the text string in the image.
        fontface: OpenCV font type
            Default cv.FONT_HERSHEY_SIMPLEX
        fontscale: float
            Font scale factor that is multiplied by the font-specific base
            size.
        color: tuple of ints
            Color of circle (r, g, b). Default (255, 100, 0)
        thickness: int
            Line thickness. Default 2.
        linetype: int
            OpenCV line type of circle boundary. Default cv2.LINE_AA

        Returns
        -------
        iterator
            Iterator that generates frames with frame numbers

        """
        for frameno, frame in enumerate(self._frames):
            yield cv.putText(frame, str(frameno + startat), org=org,
                             fontFace=fontface, fontScale=fontscale,
                             color=color, thickness=thickness,
                             lineType=linetype)
    @frameiterator
    def find_nonzero(self):
        """Yields the locations of non-zero pixels.

        Returns
        -------
        Iterator
            Iterates over a sequence of shape (N, 2) arrays, where N is the
            number of frames.

        """
        for frame in self._frames:
            idx = cv.findNonZero(frame)
            if idx is None:
                idx = np.zeros((0,2), dtype=np.uint16)
            else:
                idx = idx[:, 0, :]
            yield idx

    @frameiterator
    def morphologyex(self, morphtype='open', kernelsize=2):
        """Performs advanced morphological transformations on frames.

        Can perform advanced morphological transformations using an erosion
        and dilation as basic operations.

        In case of multi-channel images, each channel is processed
        independently.

        Parameters
        ----------
        morphtype: str
            Type of transformation. Choose from 'erode', 'dilate', 'open',
            'close', 'gradient', 'tophat', 'blackhat'. Default: 'open'.
        kernelsize: int
            Size of kernel in 1 dimension. Default 2.

        Returns
        -------
        Frames
            Iterates over sequence of transformed image frames.

        """
        morphtypes={'erode': cv.MORPH_ERODE,
                    'dilate': cv.MORPH_DILATE,
                    'open': cv.MORPH_OPEN,
                    'close': cv.MORPH_CLOSE,
                    'gradient': cv.MORPH_GRADIENT,
                    'tophat': cv.MORPH_TOPHAT,
                    'blackhat': cv.MORPH_BLACKHAT}
        morphnum = morphtypes.get(morphtype, None)
        if morphnum is None:
            raise ValueError(f'`{morphtype}` is not a valid morphtype')
        kernel = np.ones((kernelsize, kernelsize), np.uint8)
        for frame in self._frames:
            yield cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)

    @frameiterator
    def add_weighted(self, alpha, frames, beta, gamma=0):
        for frame1, frame2 in zip(self._frames, frames):
            yield cv.addWeighted(src1=frame1, alpha=alpha, src2=frame2,
                                 beta=beta, gamma=gamma)




class FramesColor(Frames):
    """An iterator that yields color frames.

    This class inherits from Frames, and hence has all its methods.

    """

    def __init__(self, nframes, width, height, color=(0, 0, 0), dtype='uint8'):
        """Creates an iterator that yields color frames.

        Parameters
        ----------
        nframes: int
            Number of frames to be produced.
        width: int
            Width of frame.
        height: int
            Height of frame.
        color:
            Fill value of frame. Default (0, 0, 0) (black).
        dtype: numpy dtype
            Dtype of frame. Default `uint8'

        Returns
        -------
        Iterator of numpy ndarrays

        """
        frame = framecolor(width=width, height=height, color=color,
                           dtype=dtype)
        frames = (frame.copy() for _ in range(nframes))
        super().__init__(frames=frames)


class FramesGray(Frames):
    """An iterator that yields gray frames.

    This class inherits from Frames, and hence has all its methods.

    """

    def __init__(self, nframes, width, height, value=0, dtype='uint8'):
        """Creates an iterator that yields gray frames.

        Parameters
        ----------
        nframes: int
            Number of frames to be produced.
        width: int
            Width of frame.
        height: int
            Height of frame.
        value:
            Fill value of frame. Default 0 (black).
        dtype: numpy dtype
            Dtype of frame. Default `uint8'

        Returns
        -------
        Iterator of numpy ndarrays

        """

        frame = framegray(width=width, height=height, value=value,
                          dtype=dtype)

        frames = (frame.copy() for _ in range(nframes))
        super().__init__(frames=frames)


class FramesNoise(Frames):
    """An iterator that yields noise frames.

    This class inherits from Frames, and hence has all its methods.

    """

    def __init__(self, nframes,width,  height, dtype='uint8'):
        """Creates an iterator that yields gray frames.

        Parameters
        ----------
        nframes: int
            Number of frames to be produced.
        width: int
            Width of frame.
        height: int
            Height of frame.
        dtype: numpy dtype
            Dtype of frame. Default `uint8'

        Returns
        -------
        Iterator of numpy ndarrays

        """

        frames = (framenoise(height=height, width=width, dtype=dtype)
                  for _ in range(nframes))
        super().__init__(frames=frames)


def framegray(width, height, value=0, dtype='uint8'):
    """Creates a gray frame.

    Parameters
    ----------
    width: int
        Width of frame.
    height: int
        Height of frame.
    value:
        Fill value of frame. Default 0 (black).
    dtype: numpy dtype
        Dtype of frame. Default `uint8'

    Returns
    -------
    numpy ndarray

    """
    return np.ones((height, width), dtype=dtype) * value


def framecolor(width, height, color=(0, 0, 0), dtype='uint8'):
    """Creates a color frame.

    Parameters
    ----------
    width: int
        Width of frame.
    height: int
        Height of frame.
    color:
        Fill value of frame. Default (0, 0, 0) (black).
    dtype: numpy dtype
        Dtype of frame. Default `uint8'

    Returns
    -------
    numpy ndarray

    """
    return np.ones((height, width, 3), dtype=dtype) * np.asanyarray(color,
                                                                    dtype=dtype)


def framenoise(width, height, dtype='uint8'):
    """Creates a noise frame.

    Parameters
    ----------
    width: int
        Width of frame.
    height: int
        Height of frame.
    dtype: numpy dtype
        Dtype of frame. Default `uint8'

    Returns
    -------
    numpy ndarray

    """

    return np.random.randint(0, 255, (height, width, 3), dtype=dtype)


def create_frameswithmovingcircle(nframes, width, height, framecolor=(0, 0, 0),
                                  circlecolor=(255, 100, 0), radius=6,
                                  thickness=2, linetype=8, dtype='uint8'):
    frames = FramesColor(nframes=nframes,  width=width, height=height,
                         color=framecolor, dtype=dtype)
    centers = zip(np.linspace(0, width, nframes),
                  np.linspace(0, height, nframes))
    return frames.draw_circles(centers, color=circlecolor, radius=radius,
                               thickness=thickness, linetype=linetype)
