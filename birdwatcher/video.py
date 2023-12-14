"""This module contains classes and functions to work with video files. It
depends on FFmpeg.

"""


from pathlib import Path

import numpy as np
import cv2 as cv

from .ffmpeg import videofileinfo, iterread_videofile, count_frames, \
    get_frame, get_frameat, extract_audio
from .frames import frameiterator
from .utils import progress


__all__ = ['VideoFileStream', 'testvideosmall']


class VideoFileStream():
    """Video stream from file.

    This class can read video frames from a file.

    Parameters
    ----------
    filepath : str of pathlib.Path
        Path to videofile.
    streamnumber : int, optional
        Video stream number to use as input. Often there is just
        one video stream present in a video file (default=0), but
        if there are more you can use this parameter to specify
        which one you want.

    Examples
    --------
    >>> import birdwatcher as bw
    >>> vfs = bw.VideoFileStream('zebrafinchrecording.mp4')
    >>> frames = vfs.iter_frames() # create frame iterator
    >>> frames.togray().tovideo('zebrafinchrecording_gray.mp4')

    """

    def __init__(self, filepath, streamnumber=0):

        self.filepath = fp = Path(filepath)
        self.streamnumber = streamnumber
        if not fp.exists():
            raise FileNotFoundError(f'"{filepath}" does not exist')
        metadata = videofileinfo(fp)
        self._formatmetadata = metadata['format']
        self._streammetadata = metadata['streams'][streamnumber]


    def __iter__(self):
        return self.iter_frames()

    def get_info(self):
        """Provides a dictionary will all kinds of video info.

        Much of it is provided by ffprobe.

        Returns
        -------
            Dictionary with info.
        """
        return {'classname': self.__class__.__name__,
                'classarguments': {'filepath': str(self.filepath),
                                  'streamnumber': self.streamnumber},
                'framewidth': self.framewidth,
                'frameheight': self.frameheight,
                'formatmetadata': self.formatmetadata,
                'streammetadata': self.streammetadata}

    @property
    def avgframerate(self):
        """Average frame rate of video stream, as reported in the metadata
        of the video file."""
        ar = self.streammetadata['avg_frame_rate']
        return np.divide(*map(int, ar.split('/')))

    @property
    def duration(self):
        """Duration of video stream in seconds, as reported in the metadata
        of the video file."""
        return float(self.streammetadata['duration'])

    @property
    def formatmetadata(self):
        """Metadata of video file format as provided by ffprobe."""
        return self._formatmetadata

    @property
    def streammetadata(self):
        """Metadata of video stream as provided by ffprobe."""
        return self._streammetadata

    @property
    def framewidth(self):
        """Width in pixels of frames in video stream."""
        return self._streammetadata['width']

    @property
    def frameheight(self):
        """height in pixels of frames in video stream."""
        return self._streammetadata['height']

    @property
    def framesize(self):
        """tuple (frame width, frame height) in pixels in video stream."""
        return (self.framewidth, self.frameheight)

    @property
    def nframes(self):
        """Number of frames in video stream as reported in the metadata
        of the video file. Note that this may not be accurate. Use
        `count_frames` to measure the actual number (may take a lot of
        time)."""
        return int(self.streammetadata['nb_frames'])

    def count_frames(self, threads=8, ffprobepath='ffprobe'):
        """Count the number of frames in video file stream.

        This can be necessary as the number of frames reported in the
        video file metadata may not be accurate. This method requires
        decoding the whole video stream and may take a lot of time. Use the
        `nframes` property if you trust the video file metadata and want
        fast results.

        Parameters
        ----------
        threads : int, default=8
            The number of threads you want to devote to decoding.
        ffprobepath : str or Path, default='ffprobe'

        Returns
        -------
        int
            The number of frames in video file.

        """
        return count_frames(self.filepath, threads=threads,
                            ffprobepath=ffprobepath)

    def extract_audio(self, outputpath=None, overwrite=False, 
                      codec='pcm_s24le', channel=None, ffmpegpath='ffmpeg', 
                      loglevel='quiet'):
        """Extract audio as wav file.

        Parameters
        ----------
        outputpath : str or pathlib.Path, optional
            Filename and path to write audio to. The default is None, which
            means the same directory and name as the video file is used, but 
            then with '.wav' extension.
        overwrite : bool, default=False
            Overwrite if audio file exists or not.
        codec : str, default='pcm_s24le'
            ffmpeg audio codec, with 24-bit pcm as default output.
        channel : int, default=None
            Channel number to extract. The default None will extract all 
            channels.
        ffmpegpath : str or pathlib.Path, optional
            Path to ffmpeg executable. Default is `ffmpeg`, which means it
            should be in the system path.
        loglevel : {'quiet', 'panic', 'fatal', 'error', 'warning', 'info', 
                    'verbose', 'debug' ,'trace'}, optional
            Level of info that ffmpeg should print to terminal. Default is 
            'quiet'.

        """
        filepath = self.filepath
        return extract_audio(filepath=filepath, outputpath=outputpath,
                             overwrite=overwrite, codec=codec, 
                             channel=channel, ffmpegpath=ffmpegpath, 
                             loglevel=loglevel)

    @frameiterator
    def iter_frames(self, startat=None, nframes=None, color=True,
                    ffmpegpath='ffmpeg', reportprogress=False):
        """Iterate over frames in video.

        Parameters
        ----------
        startat : str, optional
            If specified, start at this time point in the video file. You
            can use two different time unit formats: sexagesimal
            (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
        nframes  : int, optional
            Read a specified number of frames.
        color : bool, default=True
            Read as a color frame (3 dimensional) or as a gray frame (2
            dimensional). Color conversions occur through ffmpeg.
        ffmpegpath : str or pathlib.Path, optional
            Path to ffmpeg executable. Default is `ffmpeg`, which means it
            should be in the system path.
        reportprogress : bool, default=False

        Yields
        ------
        Frames
            Iterator that generates numpy array frames (height x width x color
            channel).

        """
        for i,frame in enumerate(iterread_videofile(self.filepath,
                                                    startat=startat,
                                                    nframes=nframes,
                                                    color=color,
                                                    ffmpegpath=ffmpegpath)):
            if reportprogress:
                progress(i, self.nframes)
            yield frame

    def get_frame(self, framenumber, color=True, ffmpegpath='ffmpeg'):
        """Get frame specified by frame sequence number.

        Note that this can take a lot of processing because the video
        has to be decoded up to that number. Specifying by time is more
        efficient (see: `get_frameat` method.

        Parameters
        ----------
        framenumber : int
            Get the frame `framenumber` from the video stream.
        color : bool, default=True
            Read as a color frame (3 dimensional) or as a gray frame (2
            dimensional).
        ffmpegpath : str or pathlib.Path, optional
            Path to ffmpeg executable. Default is `ffmpeg`, which means it
            should be in the system path.

        Returns
        -------
        numpy ndarray
            The frame at the specified framenumber.

        Example
        -------
        >>> import birdwatcher as bw
        >>> vfs = bw.testvideosmall()
        >>> frame = vfs.get_frame(500)

        """
        return get_frame(self.filepath, framenumber=framenumber,
                         color=color, ffmpegpath=ffmpegpath)

    def get_frameat(self, time, color=True, ffmpegpath='ffmpeg'):
        """Get frame at specified time.

        Parameters
        ----------
        time : str
            Get frame at a time point in the video file. You
            can use two different time unit formats: sexagesimal
            (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
        color : bool, default=True
            Read as a color frame (2 dimensional) or as a gray frame (3
            dimensional).
        ffmpegpath : str or pathlib.Path, optional
            Path to ffmpeg executable. Default is `ffmpeg`, which means it
            should be in the system path.

        Returns
        -------
        numpy ndarray
            The frame at the specified time.

        Example
        -------
        >>> import birdwatcher as bw
        >>> vfs = bw.testvideosmall()
        >>> frame = vfs.get_frameat('5.05') # at 5 sec and 50 msec
        >>> frame = vfs.get_frameat('00:00:05.05') # same thing

        """
        return get_frameat(self.filepath, time=time, color=color,
                           ffmpegpath=ffmpegpath)

    def show(self, startat=None, nframes=None, framerate=None):
        """Shows frames in a video window.

        The frames of a VideoFileStream are displayed in a seperate window.
        Press 'q' to quit the video before the end.

        Parameters
        ----------
         startat : str, optional
            If specified, start at this time point in the video file. You
            can use two different time unit formats: sexagesimal
            (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
        nframes  : int, optional
            Read a specified number of frames.
        framerate : int, optional
            The default framerate is None, which means that the average frame
            rate of video stream, as reported in the metadata, is used.

        """
        if framerate is None:
            framerate = self.avgframerate
        f = self.iter_frames(startat=startat, nframes=nframes)
        return f.show(framerate=framerate)


def testvideosmall():
    """A 20-s video of a zebra finch for testing purposes.

    Returns
    -------
    VideoFileStream
        An instance of Birdwatcher's VideoFileSteam class.

    """
    file = 'zf20s_low.mp4'
    path = Path(__file__).parent / 'testvideos' / file
    return VideoFileStream(path)


def walk_videofiles(dirpath, extension='.avi'):
    """Walks recursively over contents of `dirpath` and yield pathlib Path
    objects of videofiles, as defined by their `extension`.

    Parameters
    ----------
    dirpath : str or Path
        The top-level directory to start at.
    extension : str, default='.avi'
        Filter on this extension.

    """

    dirpath = Path(dirpath)
    if extension.startswith('.'):
        extension = extension[1:]
    for file in dirpath.rglob(f'*.{extension}'):
        yield file


## FIXME collect much more info
def videofilesduration(dirpath, extension='avi'):
    files = sorted(
        [f for f in walk_videofiles(dirpath=dirpath, extension=extension)])
    s = 0
    for i, f in enumerate(files):
        s += VideoFileStream(f).duration