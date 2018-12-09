import numpy as np
import cv2 as cv

__all__ = ['FrameIterator', 'FramesColor', 'FramesGray', 'framecolor',
           'framegray']


class FrameIterator:

    def __init__(self, frames):
        """An iterator of frames with useful methods.

        Parameters
        ----------
        frames: iterable that produces frames

        """
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

    def draw_circles(self, centers, radius=6, color=(255, 100, 0),
                     thickness=2,
                     linetype=8, shift=0):
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
            OpenCV line type of circle boundary. Default 8
        shift: int
            Number of fractional bits in the coordinates of the center and in
            the radius value. Default 0.

        Returns
        -------
        iterator
            Iterator that generates frames with circles

        """
        def iter_frames():
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
        return FrameIterator(iter_frames())


class FramesColor(FrameIterator):
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


class FramesGray(FrameIterator):
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
