"""This module contains classes for generating image frames and general
processing functionality such as measurement, drawing of labels/text and
saving as video files.

"""

import numpy as np
import cv2 as cv
from functools import wraps

__all__ = ['Frames', 'FramesColor', 'FramesGray', 'framecolor',
           'framegray']

def frameiteror(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return Frames(func(*args, **kwargs))
    return wrapper


class Frames:
    """An iterator of frames with useful methods.

    This is an important base class in Birdwatcher, as many functions and
    methods return this type or can use it as input. It also has useful methods
    for final output, such as a video file, measurement, or adding labels.

    Parameters
    ----------
    frames: iterable
        This can be anything that is iterable and that produces image frames. A
        numpy array, a VideoFile or another Frames object.

    Examples
    --------
    >>>import birdwatcher as bw
    >>>import numpy as np
    >>>noise = (np.random.randint(0, 255, (720, 1280, 3), dtype='uint8')
    ...         for i in range(250)) # 250 noise color frames at 720p
    >>>frames = bw.Frames(noise)
    >>>frames = frames.draw_framenumbers()
    >>>frames.tovideo('noisewithframenumbers.mp4', framerate=25)

    """

    def __init__(self, frames):

        self._frames = iter(frames)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._frames)


    def tovideo(self, filename, framerate, crf=17, format='mp4',
                codec='libx264', pixfmt='yuv420p', ffmpegpath='ffmpeg'):
        """Writes frames to video file.

        Parameters
        ----------
        filename: str
            Name of the videofile that should be written to
        framerate: int
            framerate of video in frames per second
        crf: int
            Value determines quality of video. Default: 17, which is high
            quality. See ffmpeg documentation.
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

    @frameiteror
    def draw_circles(self, centers, radius=6, color=(255, 100, 0),
                     thickness=2,
                     linetype=cv.LINE_AA, shift=0):
        """Creates a frame iterator that draws circles on an input frames
        iterable.

        Frames and centers should be iterables that have the same length.

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
        iterator
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

    @frameiteror
    def draw_framenumber(self, startat=0, org=(2, 25),
                         fontface=cv.FONT_HERSHEY_SIMPLEX,
                         fontscale=1, color=(200, 200, 200), thickness=2,
                         linetype=cv.LINE_AA):
        """Creates a frame iterator that draws the frame number on an input
        frames iterable.

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
    @frameiteror
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



class FramesColor(Frames):
    """An iterator that yields color frames.

    """

    def __init__(self, nframes, height, width, color=(0, 0, 0), dtype='uint8'):
        """Creates an iterator that yields color frames.

        Parameters
        ----------
        nframes: int
            Number of frames to be produced.
        height: int
            Height of frame.
        width: int
            Width of frame.
        color:
            Fill value of frame. Default (0, 0, 0) (black).
        dtype: numpy dtype
            Dtype of frame. Default `uint8'

        Returns
        -------
        Iterator of numpy ndarrays

        """
        frame = framecolor(height=height, width=width, color=color,
                           dtype=dtype)
        frames = (frame.copy() for _ in range(nframes))
        super().__init__(frames=frames)


class FramesGray(Frames):
    """An iterator that yields gray frames.

    """

    def __init__(self, nframes, height, width, value=0, dtype='uint8'):
        """Creates an iterator that yields gray frames.

        Parameters
        ----------
        nframes: int
            Number of frames to be produced.
        height: int
            Height of frame.
        width: int
            Width of frame.
        value:
            Fill value of frame. Default 0 (black).
        dtype: numpy dtype
            Dtype of frame. Default `uint8'

        Returns
        -------
        Iterator of numpy ndarrays

        """

        frame = framegray(height=height, width=width, value=value,
                          dtype=dtype)

        frames = (frame.copy() for _ in range(nframes))
        super().__init__(frames=frames)



def framegray(height, width, value=0, dtype='uint8'):
    """Creates a gray frame.

    Parameters
    ----------
    height: int
        Height of frame.
    width: int
        Width of frame.
    value:
        Fill value of frame. Default 0 (black).
    dtype: numpy dtype
        Dtype of frame. Default `uint8'

    Returns
    -------
    numpy ndarray

    """
    return np.ones((height, width), dtype=dtype) * value


def framecolor(height, width, color=(0, 0, 0), dtype='uint8'):
    """Creates a color frame.

    Parameters
    ----------
    height: int
        Height of frame.
    width: int
        Width of frame.
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


def create_frameswithmovingcircle(nframes, height, width, framecolor=(0, 0, 0),
                                  circlecolor=(255, 100, 0), radius=6,
                                  thickness=2, linetype=8, dtype='uint8'):
    frames = FramesColor(nframes=nframes, height=height, width=width,
                         color=framecolor, dtype=dtype)
    centers = zip(np.linspace(0, width, nframes),
                  np.linspace(0, height, nframes))
    return frames.draw_circles(centers, color=circlecolor, radius=radius,
                               thickness=thickness, linetype=linetype)
