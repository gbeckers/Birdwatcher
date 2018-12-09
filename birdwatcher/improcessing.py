import numpy as np
import cv2 as cv

__all__ = ['framesgray', 'framescolor', 'draw_circles']


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


def framesgray(nframes, height, width, value=0, dtype='uint8'):
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
    for i in range(nframes):
        yield frame.copy()


def framescolor(nframes, height, width, color=(0, 0, 0), dtype='uint8'):
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
    for i in range(nframes):
        yield frame.copy()


def draw_circles(frames, centers, radius=4, color=(255, 100, 0), thickness=2,
                 linetype=8, shift=0):
    """Creates a frame iterator that draws circles on an input frames iterable.

    Frames and centers should be iterables that have the same length.

    Parameters
    ----------
    frames: iterable
        Iterable that generates frames
    centers: iterable
        Iterable that generate center coordinates (x, y) of the circles
    radius: int
        Radius of circle. Default 4.
    color: tuple of ints
        Color of circle (r, g, b). Default (255, 100, 0)
    thickness: int
        Line thickness. Default 2.
    linetype: int
        OpenCV line type of circle boundary. Default 8
    shift: int
        Number of fractional bits in the coordinates of the center and in the
        radius value. Default 0.

    Returns
    -------
    iterator
        Iterator that generates frames with circles


    """

    for frame, center in zip(frames, centers):
        center = np.asanyarray(center)
        if not np.isnan(center).any():
            (x,y) = center.astype('int16')
            yield cv.circle(frame, center=(x,y), radius=radius, color=color,
                            thickness=thickness, lineType=linetype, shift=shift)
        else:
            yield frame

