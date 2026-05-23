"""This module contains classes and functions to work with video files. It
depends on FFmpeg.

"""

from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np

from .ffmpeg import (
    videofileinfo,
    iterread_videofile,
    count_frames,
    get_frame,
    get_frameat,
    extract_audio,
    detect_audio_codec,
)
from .frames import frameiterator
from .utils import progress


__all__ = ["VideoFile", "VideoFileStream", "testvideostreamsmall"]


class VideoFile:
    """Video file.

    This class provides stream and format information of a video file, and
    retrieves video and audio streams.

    Parameters
    ----------
    filepath : str of pathlib.Path
        Path to videofile.

    Examples
    --------
    >>> import birdwatcher as bw
    >>> vf = bw.VideoFile('zebrafinchrecording.mp4')
    >>> vf.streamsinfo # look at what streams are available
    >>> vfs = vf.get_videostream(0) # get video stream that has index 0

    """

    def __init__(self, filepath: str | Path) -> None:
        self._filepath = fp = Path(filepath)
        info = videofileinfo(fp)
        self._formatinfo = info["format"]
        self._streamsinfo = tuple(info["streams"])
        self._nstreams = len(self._streamsinfo)
        self._videostreamsinfo = tuple(
            stream for stream in self._streamsinfo if stream["codec_type"] == "video"
        )
        self._nvideostreams = len(self._videostreamsinfo)
        self._audiostreamsinfo = tuple(
            stream for stream in self._streamsinfo if stream["codec_type"] == "audio"
        )
        self._audiostreams = len(self._audiostreamsinfo)
        # self._videostreamindices = tuple(s['index'] for s in self._videostreamsinfo)
        # self._audiostreamindices = tuple(s['index'] for s in self._audiostreamsinfo)

    @property
    def filepath(self) -> Path:
        """Path to video file."""
        return self._filepath

    @property
    def formatinfo(self) -> Dict:
        """Metadata of video file format as provided by ffprobe."""
        return self._formatinfo

    @property
    def duration(self) -> float:
        return self._formatinfo["duration"]

    @property
    def streamsinfo(self) -> Tuple[Dict]:
        """Metadata of video streams as provided by ffprobe."""
        return self._streamsinfo

    @property
    def nstreams(self) -> int:
        """Number of video streams in video file."""
        return self._nstreams

    @property
    def nvideostreams(self) -> int:
        """Number of video streams in video file."""
        return self._nvideostreams

    @property
    def naudiostreams(self) -> int:
        """Number of audio streams in video file."""
        return self._audiostreams

    @property
    def videostreamsinfo(self) -> Tuple[Dict]:
        """List of metadata of video streams in video file."""
        return self._videostreamsinfo

    @property
    def audiostreamsinfo(self) -> Tuple[Dict]:
        """List of metadata of audio streams in video file."""
        return self._audiostreamsinfo

    # @property
    # def videostreamindices(self) -> Tuple[int]:
    #     """List of indices of video streams in video file."""
    #     return self._videostreamindices
    #
    # @property
    # def audiostreamindices(self) -> Tuple[int]:
    #     """List of indices of audio streams in video file."""
    #     return self._audiostreamindices

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.filepath}')"

    def __str__(self):
        s = self.__repr__()
        s += f"\n    duration: {self.duration}"
        s += f"\n    number of video streams: {self._nvideostreams}"
        s += f"\n    number of audio streams: {self._audiostreams}"
        return s

    def get_videostream(self, streamnumber: int = 0) -> "VideoFileStream":
        """
        Retrieves a video stream from the given file.

        This method uses the specified stream number to identify and return the
        corresponding video stream within the file. If no stream number is
        provided, the default stream (0) is used.

        Parameters
        ----------
        streamnumber : int or None, optional
            The number of the video stream to retrieve, by default 0.
            If `None`, the first video stream is used. Use the `videostreamindices`
            attribute to see which video streams are available.

        Returns
        -------
        VideoFileStream
            An object representing the requested video stream
        """
        return VideoFileStream(self._filepath, streamnumber=streamnumber)

    def get_audiocodec(self, streamnumber: int = 0) -> str:
        """

        Parameters
        ----------
        streamindex: int or None, optional
            Index of the audio stream. Note that this is the stream number as
            provided by the 'index' key of the `streamsinfo`,
            `videostreamsinfo` and `audiostreamsinfo` attributes.
            If `None`, the first audio stream is used. Use the `audiostreamindices`
            attribute to see which audio streams are available.

        Returns
        -------
        audiocodec: str

        """
        return detect_audio_codec(str(self._filepath), streamnumber=streamnumber)

    def extract_audio(
        self,
        outputpath: str | Path | None = None,
        overwrite: bool = False,
        codec: str = "copy",
        channel: int | None = None,
        ffmpegpath: str | Path = "ffmpeg",
        loglevel: str = "quiet",
        streamnumber: int = 0,
    ):
        """Extract audio to audio file.

        Parameters
        ----------
        outputpath : str or pathlib.Path, optional
            Filename and path to write audio to. The default is None, which means
            the same directoy and name as the video file is used, but then with an
            audio format extension. If you provide an outputpath, best is *not* to
            specify an audio extension, unless you are sure it is compatible with
            the audio codec in the video file. If not specified, a suitable file
            format with appropriate extension will be automatically selected.
        overwrite : bool, default=False
            Overwrite if audio file exists or not.
        codec : str, default='copy'
            ffmpeg audio codec, with as default copying codec to output. Another
            choice would be 'pcm_s24le', which is a high-quality setting, but may
            change the audio data as saved in video. It is recommended to use the
            default 'copy' to avoid the possibility of introducing artefacts, unless
            you know what you are doing.
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
        streamindex: int or None, optional
            Index of the audio stream. Note that this is the stream number as
            provided by the 'index' key of the `streamsinfo`,
            `videostreamsinfo` and `audiostreamsinfo` attributes.
            If `None`, the first audio stream is used. Use the `audiostreamindices`
            attribute to see which audio streams are available.

        """
        filepath = self._filepath
        return extract_audio(
            filepath=filepath,
            outputpath=outputpath,
            overwrite=overwrite,
            codec=codec,
            channel=channel,
            ffmpegpath=ffmpegpath,
            loglevel=loglevel,
            streamnumber=streamnumber,
        )


class VideoFileStream:
    """Video stream from file.

    This class can read video frames from a file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to videofile.
    streamnumber : int, deafult=0
        Video stream number to use as input. Often there is just
        one video stream present in a video file (default=0), but
        if there are more, use this parameter to specify
        which one you want.

    Examples
    --------
    >>> import birdwatcher as bw
    >>> vfs = bw.VideoFileStream('zebrafinchrecording.mp4')
    >>> frames = vfs.iter_frames() # create frame iterator
    >>> frames.togray().tovideo('zebrafinchrecording_gray.mp4')

    """

    def __init__(self, filepath: str | Path, streamnumber: int = 0):
        self._filepath = fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f'"{filepath}" does not exist')
        self._videofile = VideoFile(filepath)
        if self._videofile.nvideostreams == 0:
            raise ValueError(f"No video streams found in file '{filepath}'")
        if streamnumber >= self._videofile.nvideostreams:
            raise ValueError(
                f"Stream number {streamnumber} is out of range "
                f"(max={self._videofile.nvideostreams - 1})"
            )
        self._streamnumber = streamnumber
        self._streammetadata = self._videofile.videostreamsinfo[streamnumber]

    def __iter__(self):
        return self.iter_frames()

    def get_info(self):
        """Provides a dictionary will all kinds of video info.

        Much of it is provided by ffprobe.

        Returns
        -------
            Dictionary with info.
        """
        return {
            "classname": self.__class__.__name__,
            "classarguments": {
                "filepath": str(self.filepath),
                "streamnumber": self._streamnumber,
            },
            "framewidth": self.framewidth,
            "frameheight": self.frameheight,
            "streammetadata": self.streammetadata,
        }

    @property
    def filepath(self) -> Path:
        """Path to video file containing video stream."""
        return self._filepath

    @property
    def avgframerate(self):
        """Average frame rate of video stream, as reported in the metadata
        of the video file."""
        ar = self._streammetadata["avg_frame_rate"]
        return np.divide(*map(int, ar.split("/")))

    @property
    def codec(self):
        return self._streammetadata["codec_name"]

    @property
    def duration(self):
        """Duration of video stream in seconds, as reported in the metadata
        of the video file."""
        return float(self._streammetadata["duration"])

    @property
    def streammetadata(self):
        """Metadata of video stream as provided by ffprobe."""
        return self._streammetadata

    @property
    def framewidth(self):
        """Width in pixels of frames in video stream."""
        return self._streammetadata["width"]

    @property
    def frameheight(self):
        """height in pixels of frames in video stream."""
        return self._streammetadata["height"]

    @property
    def framesize(self):
        """tuple (frame width, frame height) in pixels in video stream."""
        return (self.framewidth, self.frameheight)

    @property
    def nframes(self) -> int | None:
        """Number of frames in video stream as reported in the metadata
        of the video file. Note that this may not be accurate. Use
        `count_frames` to measure the actual number (may take a lot of
        time). This info may also not be available, in which case None is returned."""
        nframes = self._streammetadata.get("nb_frames", None)
        if nframes is not None:
            nframes = int(nframes)
        return nframes

    @property
    def videofile(self) -> VideoFile:
        return self._videofile

    def __repr__(self):
        return (
            f"{self.__class__.__name__}('{self.filepath}', stream={self._streamnumber})"
        )

    def __str__(self):
        s = self.__repr__()
        s += f"\n    codec: {self.streammetadata['codec_name']}"
        s += f"\n    avgframerate: {self.avgframerate}"
        s += f"\n    duration: {self.duration}"
        s += f"\n    nb_frames: {self.nframes}"
        s += f"\n    framesize: {self.framesize}"
        return s

    def count_frames(self, threads=8, ffprobepath="ffprobe"):
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
        return count_frames(
            self.filepath,
            streamnumber=self._streamnumber,
            threads=threads,
            ffprobepath=ffprobepath,
        )

    @frameiterator
    def iter_frames(
        self,
        startat: str | None = None,
        startframe: int | None = None,
        nframes: int | None = None,
        stepsize: int | None = None,
        color: bool = True,
        ffmpegpath: str | Path = "ffmpeg",
        reportprogress: bool = False,
        loglevel: str = "quiet",
    ):
        """Iterate over frames in video.

        Parameters
        ----------
        startat : str, optional
            If specified, start at this time point in the video file. You
            can use two different time unit formats: sexagesimal
            (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
        startframe: int, optional
            If specified, start at this frame number. This parameter takes
            priority over `startat`.
        stepsize: int, optional
            Number of frames to advance per iteration.
        nframes  : int, optional
            Read a specified number of frames.
        color : bool, default=True
            Read as a color frame (3 dimensional) or as a gray frame (2
            dimensional). Color conversions occur through ffmpeg.
        ffmpegpath : str or pathlib.Path, optional
            Path to ffmpeg executable. Default is `ffmpeg`, which means it
            should be in the system path.
        reportprogress : bool, default=False
        loglevel : {'quiet', 'panic', 'fatal', 'error', 'warning', 'info',
            'verbose', 'debug' ,'trace'}, optional


        Yields
        ------
        Frames
            Iterator that generates numpy array frames (height x width x color
            channel).

        """
        for i, frame in enumerate(
            iterread_videofile(
                self.filepath,
                startat=startat,
                startframe=startframe,
                nframes=nframes,
                color=color,
                streamnumber=self._streamnumber,
                ffmpegpath=ffmpegpath,
                loglevel=loglevel,
            )
        ):
            if reportprogress:
                progress(i, self.nframes)
            if not stepsize or (i % stepsize) == 0:
                yield frame

    def get_frame(self, framenumber, color=True, ffmpegpath="ffmpeg"):
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
        >>> vfs = bw.testvideostreamsmall()
        >>> frame = vfs.get_frame(500)

        """
        return get_frame(
            self.filepath, framenumber=framenumber, color=color, ffmpegpath=ffmpegpath
        )

    def get_frameat(self, time, color=True, ffmpegpath="ffmpeg"):
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
        >>> vfs = bw.testvideostreamsmall()
        >>> frame = vfs.get_frameat('5.05') # at 5 sec and 50 msec
        >>> frame = vfs.get_frameat('00:00:05.05') # same thing

        """
        return get_frameat(
            self.filepath,
            time=time,
            color=color,
            streamnumber=self._streamnumber,
            ffmpegpath=ffmpegpath,
        )

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


def testvideostreamsmall():
    """A 20-s video of a zebra finch for testing purposes.

    Returns
    -------
    VideoFileStream
        An instance of Birdwatcher's VideoFileSteam class.

    """
    file = "zf20s_low.mp4"
    path = Path(__file__).parent / "testvideos" / file
    return VideoFileStream(path)


def walk_videofiles(dirpath, extension=".avi"):
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
    if extension.startswith("."):
        extension = extension[1:]
    for file in dirpath.rglob(f"*.{extension}"):
        yield file


## FIXME collect much more info
def videofilesduration(dirpath, extension="avi"):
    files = sorted([f for f in walk_videofiles(dirpath=dirpath, extension=extension)])
    s = 0
    for i, f in enumerate(files):
        s += VideoFileStream(f).duration
