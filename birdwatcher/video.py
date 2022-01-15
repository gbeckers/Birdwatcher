"""This module contains classes and functions to work with video files. It
depends on FFmpeg.

"""


import pathlib
import numpy as np
from .ffmpeg import videofileinfo, iterread_videofile, count_frames, \
    get_frame, get_frameat, extract_audio
from .frameprocessing import frameiterator
from .utils import progress, walk_paths

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
    def _frames(self):
        return self.iter_frames()

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
        """width in pixels of frames in video stream."""
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

        This can be necessary as the number of frames reported in th
        evideo file metadata may not be accurate. This method requires
        decoding the whole video stream and may take a lot of time. Use the
        `nframes` property if you trust the video file metadata and want
        fast results.

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

    def extract_audio(self, outputpath=None, overwrite=False):
        """Extract audio as 24-bit pcm wav file.

        Parameters
        ----------
        outputpath: str | pathlib.Path | None
            Filename and path to write audio to. Default is None, which
            means same name as video file, but then with '.wav' extension.
        overwrite: bool
            Overwrite if audio file exists or not. Default is False.

        """
        filepath = self.filepath
        return extract_audio(filepath=filepath, outputpath=outputpath,
                             overwrite=overwrite)

    @frameiterator
    def iter_frames(self, startat=None, nframes=None, color=True,
                    ffmpegpath='ffmpeg', reportprogress=False):
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
            Read as a color frame (3 dimensional) or as a gray frame (2
            dimensional). Default True.

        Returns
        -------
        Iterator
            Generates numpy array frames (Height x width x color channel).

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
        framenumber: int
            Get the frame `framenumber` from the video stream.
        color: bool
            Read as a color frame (2 dimensional) or as a gray frame (3
            dimensional). Default True.

        Example
        -------
        >>> import birdwatcher as bw
        >>> vf = bw.testvideosmall()
        >>> frame = vf.get_frame(500)

        Returns
        -------
        numpy ndarray
            A frame

        """

        return get_frame(self.filepath, framenumber=framenumber,
                         color=color, ffmpegpath=ffmpegpath)

    def get_frameat(self, time, color=True, ffmpegpath='ffmpeg'):
        """Get frame at specified time.

        Parameters
        ----------
        time: str
            Get frame at a time point in the video file. You
            can use two different time unit formats: sexagesimal
            (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
            Default is None.
        color: bool
            Read as a color frame (2 dimensional) or as a gray frame (3
            dimensional). Default True.

        Example
        -------
        >>> import birdwatcher as bw
        >>> vf = bw.testvideosmall()
        >>> frame = vf.get_frameat('5.05') # at 5 sec and 50 msec
        >>> frame = vf.get_frameat('00:00:05.05') # same thing

        Returns
        -------
        numpy ndarray
            The frame at the specified time.

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


def walk_videofiles(dirpath, extension='.avi'):
    """Walks recursively over contents of `dirpath` and yield pathlib Path
    objects of videofiles, as defined by their `extension`.

    Parameters
    ----------
    dirpath: str or Path
        The top-level directory to start at.
    extension: str
        Filter on this extension. Default: '.avi'

    """

    dirpath = pathlib.Path(dirpath)
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