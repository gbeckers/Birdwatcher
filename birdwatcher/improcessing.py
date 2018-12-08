import numpy as np
import cv2 as cv


def _emptyframecolor(height, width, color=(0, 0,0), dtype='uint8'):
    return np.ones((height, width, 3), dtype=dtype) * np.asanyarray(color,
                                                                    dtype=dtype)


def _emptyframegray(height, width, dtype='uint8'):
    return np.zeros((height, width), dtype=dtype)


def colorframes(nframes, height, width, color=(0, 0, 0), dtype='uint8'):
    frame = _emptyframecolor(height=height, width=width, color=color,
                             dtype=dtype)
    for i in range(nframes):
        yield frame.copy()


def circles(frames, centers, radius=4, color=(255, 0, 0), thickness=2,
            linetype=8, shift=0):
    """Draws circles on a sequence of frames.

    Frames and centers should be iterables that have the same length.

    Parameters
    ----------
    frames: iterable
        Iterable that generates frames
    centers: iterable
        Iterable that generate center coordinates (x, y)
    radius: int
        Radius of circle. Default 5
    color: tuple of ints
        Color of circle (r, g, b). Default (255, 0, 0)
    thickness: int
        Line thickness. Default -1.
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
        if not np.isnan(center).any():
            yield cv.circle(frame, center=center, radius=radius, color=color,
                      thickness=thickness, lineType=linetype, shift=shift)

