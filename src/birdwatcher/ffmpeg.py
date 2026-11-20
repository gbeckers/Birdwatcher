"""This is a helper module, providing an interface for video file input and
output using FFmpeg.

Most users will not use functions and classes from this module directly, but
rather through methods and functions in `video`, `frames` and `coordinatearrays`
modules. Birdwatcher uses FFmpeg for video and audio IO instead of OpenCV to
increase flexibility. This way, users can use a version of FFmpeg that supports
functionality not present in OpenCV. E.g., specific codecs, or CUDA support.
It is easier to compile or find specific versions of FFmpeg than of OpenCV.

"""

import codecs
import json
import subprocess
import sys
import re
import threading
import warnings
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Any
import numpy.typing as npt

import numpy as np

from .utils import peek_iterable


__all__ = ["arraytovideo", "supported_audio_codecs", "supported_video_codecs"]


AUDIOCODEC_TO_EXTENSION = {
    "aac": ".m4a",
    "mp3": ".mp3",
    "opus": ".opus",
    "vorbis": ".ogg",
    "flac": ".flac",
    "pcm_s16le": ".wav",
    "pcm_s24le": ".wav",
    "pcm_s32le": ".wav",
    "pcm_f32le": ".wav",
    "pcm_f64le": ".wav",
    "pcm_alaw": ".wav",
    "pcm_mulaw": ".wav",
    "alac": ".m4a",
    "eac3": ".eac3",
    "ac3": ".ac3",
    "dts": ".dts",
    "truehd": ".thd",
    "mp2": ".mp2",
    "wmav2": ".wma",
    "wmav1": ".wma",
}


class FFmpegError(RuntimeError):
    def __init__(
        self, cmd: list[str], returncode: int, stdout: str = "", stderr: str = ""
    ) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

        message = (
            f"FFmpeg command failed with exit code {returncode}\n"
            f"Command: {' '.join(map(str, cmd))}"
        )

        if stderr:
            message += f"\n\nstderr:\n{stderr.strip()}"

        super().__init__(message)


class NoAudioStreamError(ValueError):
    pass


def run_ffmpeg(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True, **kwargs)

    if result.returncode != 0:
        raise FFmpegError(
            cmd=args,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    return result


def run_ffprobe_json(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise FFmpegError(
            cmd=args,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    return json.loads(result.stdout)


def _effective_loglevel(loglevel: str) -> str:
    """Returns the loglevel to pass to ffmpeg.

    FFmpeg's stderr is always captured, so that its error messages can be
    included in an `FFmpegError`. A user-requested 'quiet' therefore runs
    ffmpeg at 'error' level; that output is captured but not echoed.

    """
    _check_loglevelarg(loglevel)
    return "error" if loglevel == "quiet" else loglevel


def _echo_stderr(text: str, loglevel: str) -> None:
    """Prints captured ffmpeg stderr output, unless loglevel is 'quiet'."""
    if text and loglevel != "quiet":
        sys.stderr.write(text)
        sys.stderr.flush()


class _StderrReader(threading.Thread):
    """Drains a subprocess's stderr in a background thread.

    Without this, ffmpeg can block on a full stderr pipe while we block on
    reading its stdout or writing its stdin, which deadlocks both processes.
    Output is optionally echoed to sys.stderr, and the last `maxbytes` are
    kept for error reporting.

    """

    def __init__(self, stream, loglevel: str, maxbytes: int = 65536) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._loglevel = loglevel
        self._maxbytes = maxbytes
        self._buffer = bytearray()
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def run(self) -> None:
        while chunk := self._stream.read1(4096):
            _echo_stderr(self._decoder.decode(chunk), self._loglevel)
            self._buffer += chunk
            if len(self._buffer) > self._maxbytes:
                del self._buffer[: -self._maxbytes]

    def text(self) -> str:
        """Waits until stderr is closed and returns the captured output."""
        self.join()
        return self._buffer.decode("utf-8", errors="replace")


def arraytovideo(
    frames,
    filepath: str | Path,
    framerate: int,
    scale: tuple | None = None,
    crf: int = 17,
    vformat: str = "mp4",
    codec: str = "libopenh264",
    pixfmt: str = "yuv420p",
    ffmpegpath: str | Path = "ffmpeg",
    loglevel: str = "quiet",
    overwrite: bool = False,
) -> Path:
    """Writes an iterable of numpy frames as video file using ffmpeg.

    Parameters
    ----------
    frames : iterable
        Iterable should produce numpy height x width x channel arrays with
        values ranging from 0 to 255. Frames can be color (3-dim) or gray
        (2-dim).
    filepath : str
        Name of the videofilepath that should be written to.
    framerate : int
        Framerate of video in frames per second.
    scale : tuple, optional
        (width, height). The default (None) does not change width and height.
    crf : int, default=17
        Value determines quality of video. The default 17 is high quality.
        Use 23 for good quality.
    vformat : str, default='mp4'
        ffmpeg video format.
    codec : str, default='libopenh264'
        ffmpeg video codec.
    pixfmt : str, default='yuv420p'
        ffmpeg pixel format.
    ffmpegpath : str or pathlib.Path, optional
        Path to ffmpeg executable. Default is `ffmpeg`, which means it
        should be in the system path.
    loglevel : {'quiet', 'panic', 'fatal', 'error', 'warning', 'info',
               'verbose', 'debug' ,'trace'}, optional
        Level of info that ffmpeg should print to terminal. Default is
        'quiet'.
    overwrite : bool, default=False

    Notes
    -----
    See ffmpeg documentation for more information.

    """
    ffmpegloglevel = _effective_loglevel(loglevel)
    frame, framegen = peek_iterable(frames)
    height, width, *_ = frame.shape
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise IOError(
            f'file "{filepath}" already exists, use the `overwrite` '
            f"parameter to overwrite"
        )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if frame.ndim == 2:
        ipixfmt = "gray"
    elif frame.ndim == 3:
        if frame.shape[2] == 3:
            ipixfmt = "bgr24"
        else:
            raise ValueError(
                f"Last dimension of color frame should be length 3, "
                f"got length {frame.shape[2]}"
            )
    else:
        raise ValueError(f"Frames must be 2D or 3D arrays, got {frame.ndim}D array")
    args = [
        str(ffmpegpath),
        #'-hwaccel',
        "-loglevel",
        ffmpegloglevel,
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        ipixfmt,
        "-r",
        f"{framerate}",
        "-s",
        f"{width}x{height}",
        "-i",
        "pipe:",
    ]
    if codec is not None:
        args += ["-vcodec", f"{codec}"]
    if vformat is not None:
        args += ["-f", f"{vformat}"]
    if crf is not None:
        args += ["-crf", f"{crf}"]
    if pixfmt is not None:
        args += ["-pix_fmt", f"{pixfmt}"]
    if scale is not None:
        outwidth, outheight = scale
        args.extend(["-vf", f"scale={outwidth}:{outheight}"])
    args.extend(["-y", str(filepath)])
    p = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stderrreader = _StderrReader(p.stderr, loglevel=loglevel)
    stderrreader.start()
    try:
        pipebroken = False
        for frame in framegen:
            if frame.dtype != np.uint8:
                raise TypeError("Frames must have dtype uint8")
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            try:
                p.stdin.write(frame.tobytes())
            except OSError:
                # ffmpeg exited early (BrokenPipeError on POSIX, EINVAL on
                # Windows); its own error message is reported below.
                pipebroken = True
                break
        try:
            p.stdin.close()
        except OSError:
            pipebroken = True
        p.wait()
        if p.returncode != 0 or pipebroken:
            raise FFmpegError(
                cmd=args,
                returncode=p.returncode,
                stderr=stderrreader.text(),
            )
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        stderrreader.join()
    return Path(filepath)


def videofileinfo(
    filepath: str | Path, ffprobepath: str | Path = "ffprobe"
) -> dict[str, Any]:
    args = [
        str(ffprobepath),
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(filepath),
    ]
    return run_ffprobe_json(args)


def ffmpegversion(ffmpegpath: str | Path = "ffmpeg") -> str | None:
    args = [str(ffmpegpath), "-version"]
    result = run_ffmpeg(args)
    firstline = result.stdout.split("\n")[0]
    match = re.search(r"ffmpeg version (\S+)", firstline)
    if match:
        return match.group(1)
    else:
        return None


def iterread_videofile(
    filepath: str | Path,
    startat: str | None = None,
    startframe: int | None = None,
    nframes: int | None = None,
    color: bool = True,
    streamnumber: int = 0,
    ffmpegpath: str | Path = "ffmpeg",
    loglevel: str = "quiet",
) -> Iterator[npt.NDArray[np.uint8]]:
    """
    Parameters
    ----------
    filepath : str
        Name of the videofilepath.
    startat : str
      There are two accepted syntaxes for expressing time duration.
      [-][HH:]MM:SS[.m...], where HH expresses the number of hours,
      MM the number of minutes for a maximum of 2 digits, and SS
      the number of seconds for a maximum of 2 digits. The m at the
      end expresses decimal value for SS.
      [-]S+[.m...][s|ms|us], where S expresses the number of seconds,
      with the optional decimal part m. The optional literal suffixes
      ‘s’, ‘ms’ or ‘us’ indicate to interpret the value as seconds,
      milliseconds or microseconds, respectively. The following
      examples are all valid time duration: ‘55’ means 55 seconds,
      ‘0.2’ means 0.2 seconds, ‘200ms’ means 200 milliseconds,
      ‘200000us’ means 200000 microseconds, ‘12:03:45’ means 12 hours,
      03 minutes and 45 seconds, ‘23.189’ means 23.189 seconds.
    startframe : int or None, default=None
      If specified, overrides startat.
    nframes : int
    color : bool, default=True
      Return color frames (True) or gray frames (False).
    streamnumber : int, default=0
      Which video stream to read.
    ffmpegpath : str or pathlib.Path, optional
        Path to ffmpeg executable. Default is `ffmpeg`, which means it
        should be in the system path.
    loglevel : {'quiet', 'panic', 'fatal', 'error', 'warning', 'info',
               'verbose', 'debug' , 'trace'}, optional
        Level of info that ffmpeg should print to terminal. Default is
        'quiet'.

    Yields
    -------
    numpy ndarray
        Generates numpy arrays of video frames.

    """
    # frameshape, framesize, frameheight, framewidth, pix_fmt
    ffmpegloglevel = _effective_loglevel(loglevel)
    frameproperties = _get_frameproperties(
        filepath=filepath, color=color, streamnumber=streamnumber
    )
    args = [str(ffmpegpath), "-loglevel", ffmpegloglevel]

    if startframe is not None:  # overrides startat
        args.extend(
            [
                "-i",
                str(filepath),
                "-map",
                f"0:v:{streamnumber}",
                "-vf",
                f"select=gte(n\\,{startframe})",
            ]
        )
    elif startat is not None:
        args.extend(
            ["-i", str(filepath), "-map", f"0:v:{streamnumber}", "-ss", startat]
        )
    else:
        args.extend(["-i", str(filepath), "-map", f"0:v:{streamnumber}"])
    if nframes is not None:
        args += ["-vframes", str(nframes)]
    args += [
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        frameproperties.pix_fmt,
        "-f",
        "rawvideo",
        "pipe:1",
    ]

    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderrreader = _StderrReader(p.stderr, loglevel=loglevel)
    stderrreader.start()
    try:
        frameno = 0
        while True:
            if nframes is not None and frameno >= nframes:
                break
            data = p.stdout.read(frameproperties.size)
            if len(data) < frameproperties.size:
                if len(data) > 0:
                    warnings.warn(
                        f"Incomplete frame at position {frameno} "
                        f"({len(data)} bytes, expected {frameproperties.size}). "
                        "The video may be truncated."
                    )
                # Natural EOF: surface ffmpeg failures so callers see a real
                # error instead of a silently empty/truncated iterator.
                p.wait()
                if p.returncode != 0:
                    raise FFmpegError(
                        cmd=args,
                        returncode=p.returncode,
                        stderr=stderrreader.text(),
                    )
                break
            yield np.frombuffer(data, dtype=np.uint8).reshape(frameproperties.shape)
            frameno += 1
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        p.stdout.close()
        stderrreader.join()


def count_frames(
    filepath: str | Path,
    streamnumber: int = 0,
    threads: int = 8,
    ffprobepath: str | Path = "ffprobe",
) -> int:
    args = [
        str(ffprobepath),
        "-threads",
        str(threads),
        "-count_frames",
        "-select_streams",
        f"v:{streamnumber}",
        "-show_entries",
        "stream=nb_read_frames",
        "-print_format",
        "json",
        str(filepath),
    ]
    result = run_ffprobe_json(args)
    return int(result["streams"][0]["nb_read_frames"])


def get_frame(
    filepath: str | Path,
    framenumber: int,
    color: bool = True,
    streamnumber: int = 0,
    ffmpegpath: str | Path = "ffmpeg",
    loglevel: str = "quiet",
) -> npt.NDArray[np.uint8]:
    ffmpegloglevel = _effective_loglevel(loglevel)
    frameproperties = _get_frameproperties(
        filepath=filepath, color=color, streamnumber=streamnumber
    )
    args = [
        str(ffmpegpath),
        "-loglevel",
        ffmpegloglevel,
        "-i",
        str(filepath),
        "-map",
        f"0:v:{streamnumber}",
    ]
    args += [
        "-vcodec",
        "rawvideo",
        "-vf",
        f"select='eq(n\\,{framenumber})'",
        "-vframes",
        "1",
        "-pix_fmt",
        frameproperties.pix_fmt,
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    # subprocess.run reads stdout and stderr concurrently, so ffmpeg can't
    # block on a full stderr pipe.
    result = subprocess.run(args, capture_output=True)
    stderr = result.stderr.decode("utf-8", errors="replace")
    _echo_stderr(stderr, loglevel)
    data = result.stdout
    if result.returncode != 0 or len(data) != frameproperties.size:
        raise FFmpegError(cmd=args, returncode=result.returncode, stderr=stderr)
    return np.frombuffer(data, dtype=np.uint8).reshape(frameproperties.shape)


def get_frameat(
    filepath: str | Path,
    time: str,
    color: bool = True,
    streamnumber: int = 0,
    ffmpegpath: str | Path = "ffmpeg",
    loglevel: str = "quiet",
) -> npt.NDArray[np.uint8]:
    return next(
        iterread_videofile(
            filepath,
            startat=time,
            nframes=1,
            color=color,
            streamnumber=streamnumber,
            ffmpegpath=ffmpegpath,
            loglevel=loglevel,
        )
    )


@dataclass
class FrameProperties:
    shape: tuple[int, ...]
    size: int
    width: int
    height: int
    pix_fmt: str


def _get_frameproperties(
    filepath: str | Path, color: bool, streamnumber: int = 0
) -> FrameProperties:
    """Convenience function that produces frame characteristics for a given
    video stream. Handy if you want to know the format of a frame that is returned
    by ffmpeg from a pipe."""
    vsi = videofileinfo(filepath)["streams"]
    videostreamsinfo = tuple(
        stream for stream in vsi if stream["codec_type"] == "video"
    )
    if not 0 <= streamnumber < len(videostreamsinfo):
        raise ValueError(
            f'No video stream number {streamnumber} in "{filepath}" '
            f"({len(videostreamsinfo)} video stream(s) available)"
        )
    height = videostreamsinfo[streamnumber]["height"]
    width = videostreamsinfo[streamnumber]["width"]
    if color:
        shape = (height, width, 3)
        size = height * width * 3
        pix_fmt = "bgr24"
    else:
        shape = (height, width)
        size = height * width
        pix_fmt = "gray"
    return FrameProperties(
        shape=shape, size=size, width=width, height=height, pix_fmt=pix_fmt
    )


def _check_loglevelarg(loglevelarg: str) -> None:
    levels = (
        "quiet",
        "panic",
        "fatal",
        "error",
        "warning",
        "info",
        "verbose",
        "debug",
        "trace",
    )
    if loglevelarg not in levels:
        raise ValueError(
            f"`loglevel` argument ('{loglevelarg}') should be one of: {levels}"
        )


# Audio


def _audiostreaminfo(
    filepath: str | Path,
    streamnumber: int = 0,
    ffprobepath: str | Path = "ffprobe",
) -> dict[str, Any] | None:
    """Returns ffprobe info of an audio stream, or None if it doesn't exist."""
    args = [
        str(ffprobepath),
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        f"a:{streamnumber}",
        str(filepath),
    ]
    streams = run_ffprobe_json(args).get("streams", [])
    return streams[0] if streams else None


def detect_audio_codec(
    filepath: str | Path,
    streamnumber: int = 0,
    ffprobepath: str | Path = "ffprobe",
) -> str | None:
    """Detect the audio codec used in an audio stream of a video file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the video file.
    streamnumber : int, default=0
        Which audio stream to inspect.
    ffprobepath : str or pathlib.Path, optional
        Path to ffprobe executable. Default is `ffprobe`, which means it should
        be in the system path.

    Returns
    -------
    str or None
        Codec name, or None if there is no such audio stream.

    """
    info = _audiostreaminfo(filepath, streamnumber, ffprobepath)
    return None if info is None else info["codec_name"]


def supported_encoders(kind: str, ffmpegpath: str | None = "ffmpeg") -> set[str]:
    """
    Checks which encoders are supported by FFmpeg for encoding/writing.

    Parameters
    ----------
    kind: str
        Kind of encoder to check, "A" = audio, "V" = video, "S" = subtitle
    ffmpegpath : str
        Path to ffmpeg executable

    Returns
    -------
    set of codecs

    """

    _ENCODER_RE = re.compile(
        rf"^\s*{re.escape(kind)}\S*\s+([^\s]+)",
        re.MULTILINE,
    )

    args = [str(ffmpegpath), "-hide_banner", "-encoders"]
    result = run_ffmpeg(args)
    encoders = {
        match.group(1).lower() for match in _ENCODER_RE.finditer(result.stdout)
    }
    encoders.discard("=")
    return encoders


supported_audio_codecs = functools.partial(supported_encoders, "A")

supported_video_codecs = functools.partial(supported_encoders, "V")


def extract_audio(
    filepath: str | Path,
    outputpath: str | Path | None = None,
    overwrite: bool = False,
    codec: str = "copy",
    channel: int | None = None,
    streamnumber: int = 0,
    ffmpegpath: str | Path = "ffmpeg",
    loglevel: str = "quiet",
    ffprobepath: str | Path = "ffprobe",
) -> Path:
    """Extract audio as wav file.

    Parameters
    ----------
    filepath : str or pathlib.Path
      Name of the video file path.
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
        Channel number to extract, counting from 1 (so 1 is the first
        channel). The default None will extract all channels. Extracting a
        single channel requires re-encoding, so it cannot be combined with
        `codec='copy'`; specify an audio codec such as 'pcm_s24le'.
    streamnumber : int, default=0
        Which audio stream to extract.
    ffmpegpath : str or pathlib.Path, optional
        Path to ffmpeg executable. Default is `ffmpeg`, which means it should
        be in the system path.
    loglevel : {'quiet', 'panic', 'fatal', 'error', 'warning', 'info',
                'verbose', 'debug' ,'trace'}, optional
        Level of info that ffmpeg should print to terminal. Default is
        'quiet'.
    ffprobepath : str or pathlib.Path, optional
        Path to ffprobe executable. Default is `ffprobe`, which means it should
        be in the system path.

    Returns
    -------
    outputpath: Path
        The path of the generated audio file

    """
    filepath = Path(filepath)
    ffmpegloglevel = _effective_loglevel(loglevel)
    if channel is not None:
        if codec == "copy":
            raise ValueError(
                "Extracting a single channel requires re-encoding, which is "
                "not possible with codec='copy'. Specify an audio codec, "
                "e.g. codec='pcm_s24le'."
            )
        if channel < 1:
            raise ValueError(
                f"`channel` should be 1 or higher (channels are numbered "
                f"from 1), got {channel}"
            )
    streaminfo = _audiostreaminfo(
        filepath, streamnumber=streamnumber, ffprobepath=ffprobepath
    )
    if streaminfo is None:
        raise NoAudioStreamError(
            f'No audio stream number {streamnumber} in file "{filepath}"'
        )
    inputcodec = streaminfo["codec_name"]
    nchannels = streaminfo.get("channels")
    # ffmpeg's pan filter silently outputs silence for nonexisting channels
    if channel is not None and nchannels is not None and channel > nchannels:
        raise ValueError(
            f"`channel` is {channel}, but audio stream {streamnumber} has "
            f"only {nchannels} channel(s)"
        )
    # The codec name we look up to choose an output extension is not the
    # same as the codec we pass to ffmpeg. When the user asks for stream
    # copy, we still need the input's codec name to pick a sensible
    # extension, but ffmpeg must receive "copy" so that no re-encoding
    # happens.
    if codec == "copy":
        codec_for_ext = inputcodec
    else:
        if codec not in supported_audio_codecs(ffmpegpath=ffmpegpath):
            raise ValueError(f'ffmpeg does not support codec "{codec}"')
        codec_for_ext = codec
    if outputpath is None:
        outputpath = Path(filepath).with_suffix("")
    ext = AUDIOCODEC_TO_EXTENSION.get(codec_for_ext)
    outputpath = Path(outputpath)
    if outputpath.suffix:
        if ext is not None and outputpath.suffix.lower() != ext:
            warnings.warn(
                f'Specified audio extension ("{outputpath.suffix}") is '
                f'not the same as the default ("{ext}") for this codec '
                f'("{codec_for_ext}"). Proceeding anyway.'
            )
    elif ext is not None:
        outputpath = outputpath.with_suffix(ext)
    else:
        outputpath = outputpath.with_suffix(".mkv")  # fallback Matroska
    if outputpath.exists() and not overwrite:
        raise FileExistsError(
            f'"{outputpath}" already exists, use `overwrite` parameter'
        )
    args = [
        str(ffmpegpath),
        "-loglevel",
        ffmpegloglevel,
        "-y",
        "-i",
        str(filepath),
        "-map",
        f"0:a:{streamnumber}",
        "-vn",
        "-c:a",
        codec,
    ]
    if channel is not None:
        args += ["-af", f"pan=mono|c0=c{channel - 1}"]
    args += [str(outputpath)]
    result = run_ffmpeg(args)
    _echo_stderr(result.stderr, loglevel)
    return outputpath
