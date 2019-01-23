"""This module contains classes and functions to work with video files. It
depends on FFmpeg.

"""


import pathlib
import numpy as np
from .ffmpeg import videofileinfo, iterread_videofile, count_frames, \
    get_frameat
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
        Video stream number to use as input. Often there is just
        one video stream present in a video file, but if there are more you
        can use this parameter to specify which one you want. Default 0.

    Examples
    --------
    >>> import birdwatcher as bw
    >>> vf = bw.VideoFileStream('zebrafinchrecording.mp4')
    >>> frames = vf.iter_frames() # create frame iterator
    >>> frames.togray().tovideo('zebrafinchrecording_gray.mp4')

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

    def get_info(self):
        return {'classname': self.__class__.__name__,
                'classarguments': {'filepath': str(self.filepath),
                                  'streamnumber': self.streamnumber},
                'framewidth': self.framewidth,
                'frameheight': self.frameheight,
                'formatmetadata': self.formatmetadata,
                'streammetadata': self.streammetadata}

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

    @property
    def framesize(self):
        return (self.framewidth, self.frameheight)

    @property
    def avgframerate(self):
        """Average frame rate of video stream"""
        ar = self.streammetadata['avg_frame_rate']
        return np.divide(*map(int, ar.split('/')))

    def count_frames(self, threads=8, ffprobepath='ffprobe'):
        """Count the number of frames in video file stream.

        This requires decoding the whole video because th enumber of frames
        specified in the metadata may not be accurate.

        Parameters
        ----------
        threads: int
            The number of threads you want to devote to decoding.

        ffprobepath: str or Path

        Returns
        -------
        int
            The number of frames in video file

        """
        return count_frames(self.filepath, threads=threads,
                            ffprobepath=ffprobepath)

    @frameiterator
    def iter_frames(self, startat=None, nframes=None, color=True,
                    ffmpegpath='ffmpeg'):
        """Iterate over frames in video.

        Parameters
        ----------
        startat: str or None
            If specified, start at this time point in the video file. You
            can use two different time unit formats: sexagesimal
            (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
            Default is None.
        nframes: int
            Read a specified number of frames.
        color: bool
            Read as a color frame (2 dimensional) or as a gray frame (3
            dimensional). Default True.

        Returns
        -------
        Iterator
            Generates numpy array frames (Height x width x color channel).

        """
        return iterread_videofile(self.filepath, startat=startat,
                                  nframes=nframes, color=color,
                                  ffmpegpath=ffmpegpath)

    def get_frameat(self, time, color=True, ffmpegpath='ffmpeg'):
        """Get frame at specified time.

        Parameters
        ----------
        time: str or None
            If specified, start at this time point in the video file. You
            can use two different time unit formats: sexagesimal
            (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
            Default is None.
        color: bool
            Read as a color frame (2 dimensional) or as a gray frame (3
            dimensional). Default True.

        Returns
        -------

        """
        return get_frameat(self.filepath, time=time, color=color,
                           ffmpegpath=ffmpegpath)


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