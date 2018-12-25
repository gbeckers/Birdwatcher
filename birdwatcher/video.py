"""This module contains classes and functions to work with video files. It
depends on FFmpeg.

"""


import pathlib
from .ffmpeg import videofileinfo, iterread_videofile
from .frameprocessing import frameiterator

__all__ = ['VideoFileStream', 'testvideosmall']

class VideoFileStream():
    """Video stream from file.

        This class can read video frames from a file.
        Parameters
        ----------
        filepath: str of pathlib.Path
            Path to videofile.
        streamnumber: int
            Stream number to use as input. Default 1.

        Examples
        --------
        >>> import birdwatcher as bw
        >>> vf = bw.VideoFile('zebrafinchrecording.mp4')
        >>> vf.togray().tovideo('zebrafinchrecording_gray.mp4')
        >>> vf.get_framebynumber(100).max()
        255

        """

    def __init__(self, filepath, streamnumber=0):
        self.filepath = fp = pathlib.Path(filepath)
        self.streamnumber = streamnumber
        if not fp.exists():
            raise FileNotFoundError(f'"{filepath}" does not exist')
        metadata = videofileinfo(fp)
        self._formatmetadata = metadata['format']
        self._streammetadata = metadata['streams'][streamnumber]


    def __iter__(self):
        return self.iter_frames()

    @property
    def _frames(self):
        return self.iter_frames()

    @property
    def formatmetadata(self):
        return self._formatmetadata

    @property
    def streammetadata(self):
        return self._streammetadata

    @property
    def framewidth(self):
        return self._streammetadata['width']

    @property
    def frameheight(self):
        return self._streammetadata['height']

    @frameiterator
    def iter_frames(self, stopframe=None, color=True, pix_fmt='bgr24',
                    ffmpegpath='ffmpeg'):
        """Iterate over frames in video.

        Parameters
        ----------
        stopframe: int
            Stop at frame `stopframe`

        Returns
        -------
        Iterator
            Generates numpy array frames (Height x width x color channel).

        """
        return iterread_videofile(self.filepath, stopframe=stopframe,
                                  color=color, pix_fmt=pix_fmt,
                                  ffmpegpath=ffmpegpath)

    def derive_filepath(self, append_string='', suffix=None, path=None):
        """Generate a file path based on the name and potentially path of the
        video.

        Parameters
        ----------
        append_string: str
            String to append to file name stem. Default: ''.
        suffix: str or None
            File extension to use. If None, the same as video file.
        path: str or pathlib.Path or None
            Path to use. If None use same path as video file.

        Returns
        -------
        pathlib.Path
            Path derived from video file path.

        """
        stem = self.filepath.stem
        if suffix is None:
            suffix = self.filepath.suffix
        filename = f'{stem}_{append_string}{suffix}'
        if path is None:
            dpath = self.filepath.parent / filename
        else:
            dpath = pathlib.Path(path) / filename
        return dpath

def testvideosmall():
    """A 20-s video of a zebra finch for testing purposes.

    Returns
    -------
    VideoFile
        An instance of Birdwatcher's VideoFile class.

    """
    file = 'zf20s_low.mp4'
    path = pathlib.Path(__file__).parent / 'testvideos' / file
    return VideoFileStream(path)