"""This module provides an interface for video file input and output using
FFmpeg. Most users will not use functions and classes from this module
directly, but rather through methods and functions in other modules. We use
FFmpeg instead of video IO in OpenCV because we want to be more flexible so
that the user can decide for a (potentially self-compiled) version of FFmpeg
that supports functionality not present in OpenCV. E.g., specific codecs,
CUDA support. It is easier to compile or find specific versions of FFmpeg
than of OpenCV. Keeping all video IO in this module, allows for the easy
addition of other ways of video IO in Birdwatcher in the future.

"""

import json
import subprocess
import re
import warnings
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Generator
from numpy.typing import NDArray

import numpy as np

from .utils import peek_iterable


__all__ = ['arraytovideo', 'supported_audio_codecs', 'supported_video_codecs']


AUDIOCODEC_TO_EXTENSION = {
    "aac":        ".m4a",
    "mp3":        ".mp3",
    "opus":       ".opus",
    "vorbis":     ".ogg",
    "flac":       ".flac",
    "pcm_s16le":  ".wav",
    "pcm_s24le":  ".wav",
    "pcm_s32le":  ".wav",
    "pcm_f32le":  ".wav",
    "pcm_alaw":   ".wav",
    "pcm_mulaw":  ".wav",
    "alac":       ".m4a",
    "eac3":       ".eac3",
    "ac3":        ".ac3",
    "dts":        ".dts",
    "truehd":     ".thd",
    "mp2":        ".mp2",
    "wmav2":      ".wma",
    "wmav1":      ".wma",
}


class FFmpegError(RuntimeError):
    def __init__(self, cmd, returncode, stdout="", stderr=""):
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


def run_ffmpeg(args: list, **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        **kwargs)

    if result.returncode != 0:
        raise FFmpegError(
            cmd=args,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    return result


def run_ffprobe_json(args: list) -> dict:
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


def arraytovideo(frames, filepath: str | Path, framerate: int,
                 scale: tuple | None = None, crf: int = 17, vformat: str = 'mp4',
                 codec: str = 'libopenh264', pixfmt: str = 'yuv420p',
                 ffmpegpath: str | Path = 'ffmpeg', loglevel: str = 'quiet',
                 overwrite: bool = False) -> Path:
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
    _check_loglevelarg(loglevel)
    frame, framegen = peek_iterable(frames)
    height, width, *_ = frame.shape
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise IOError(
            f'file "{filepath}" already exists, use the `overwrite` '
            f'parameter to overwrite' )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if frame.ndim == 2:
        ipixfmt = 'gray'
    elif frame.ndim == 3:
        if frame.shape[2] == 3:
            ipixfmt = 'bgr24'
        else:
            raise ValueError(
                f"Last dimension of color frame should be length 3, "
                f"got length {frame.shape[2]}"
            )
    else:
        raise ValueError(
            f"Frames must be 2D or 3D arrays, got {frame.ndim}D array"
        )
    args = [str(ffmpegpath),
            #'-hwaccel',
            '-loglevel' , loglevel,
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-pix_fmt', ipixfmt,
            '-r', f'{framerate}',
            '-s', f'{width}x{height}', '-i', 'pipe:']
    if codec is not None:
        args += ['-vcodec', f'{codec}']
    if vformat is not None:
        args += ['-f', f'{vformat}']
    if crf is not None:
        args += ['-crf', f'{crf}']
    if pixfmt is not None:
        args +=['-pix_fmt', f'{pixfmt}']
    if scale is not None:
        outwidth, outheight = scale
        args.extend(['-vf', f'scale={outwidth}:{outheight}'])
    args.extend(['-y', str(filepath)])
    p = None
    try:
        p = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for frame in framegen:
            if frame.dtype != np.uint8:
                raise TypeError("Frames must have dtype uint8")
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            p.stdin.write(frame.tobytes())
        p.stdin.close()
        p.stdin = None
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise FFmpegError(
                cmd=args,
                returncode=p.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )
    except Exception:
        if p is not None:
            p.kill()
            p.wait()
        raise
    return Path(filepath)


def videofileinfo(filepath: str | Path, ffprobepath: str | Path = 'ffprobe') -> dict:
    args = [str(ffprobepath), '-print_format', 'json', '-show_format',
            '-show_streams', str(filepath)]
    return run_ffprobe_json(args)


def ffmpegversion(ffmpegpath: str | Path ='ffmpeg') -> str | None:
    args = [str(ffmpegpath), '-version']
    result = run_ffmpeg(args)
    firstline =  result.stdout.split("\n")[0]
    match = re.search(r"ffmpeg version (\S+)", firstline)
    if match:
        return match.group(1)
    else:
        return None

## FIXME inform before raising StopIteration that file has no frames
## FIXME startat can be precise or not
def iterread_videofile(filepath: str | Path, startat: str | None = None,
                       nframes: int | None = None, color: bool=True,
                       streamnumber: int = 0, ffmpegpath: str | Path = 'ffmpeg',
                       loglevel: str = 'quiet') -> Generator[NDArray, None, None]:
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
    nframes : int
    color : bool, default=True
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
    _check_loglevelarg(loglevel)
    frameproperties = \
        _get_frameproperties(filepath=filepath, color=color)
    args = [str(ffmpegpath), '-loglevel', loglevel]
    if startat is not None:
        args.extend(['-i', str(filepath), '-map', f'0:v:{streamnumber}',
                     '-ss', startat])
    else:
        args.extend(['-i', str(filepath)])
    if nframes is not None:
        args += ['-vframes', str(nframes)]
    args += ['-c:v', 'rawvideo', '-pix_fmt', frameproperties.pix_fmt,
             '-f', 'rawvideo', 'pipe:1']

    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        frameno = 0
        while True:
            data = p.stdout.read(frameproperties.size)
            if len(data) < frameproperties.size:
                if len(data) > 0:
                    warnings.warn(
                        f"Incomplete frame at position {frameno} "
                        f"({len(data)} bytes, expected {frameproperties.size}). "
                        "The video may be truncated."
                    )
                break
            if nframes is not None and frameno >= nframes:
                break
            yield np.frombuffer(data, dtype=np.uint8).reshape(frameproperties.shape)
            frameno += 1
    finally:
        p.kill()
        p.wait()

#TODO check threads code
def count_frames(filepath: str | Path, streamnumber: int = 0,
                 threads: int = 8, ffprobepath: str | Path = 'ffprobe') -> int:
    args = [str(ffprobepath), '-threads:0', str(threads),
            '-count_frames', '-select_streams', f'v:{streamnumber}', '-show_entries',
            'stream=nb_read_frames', '-print_format', 'json', str(filepath)]
    result = run_ffprobe_json(args)
    return int(result['streams'][0]['nb_read_frames'])


def get_frame(filepath: str | Path, framenumber: int, color: bool = True,
              streamnumber: int = 0, ffmpegpath: str | Path = 'ffmpeg',
              loglevel: str = 'quiet') -> NDArray:
    _check_loglevelarg(loglevel)
    frameproperties = _get_frameproperties(filepath=filepath, color=color)
    args = [str(ffmpegpath), '-loglevel' , loglevel, '-i', str(filepath),
            '-map', f'0:v:{streamnumber}']
    args +=['-vcodec', 'rawvideo',  '-vf', f"select='eq(n\\,{framenumber})'",
            '-vframes', '1', '-pix_fmt', frameproperties.pix_fmt,
            '-f', 'rawvideo', 'pipe:1']
    with subprocess.Popen(args, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as p:
        return np.frombuffer(p.stdout.read(frameproperties.size),
                             dtype=np.uint8).reshape(frameproperties.shape)


def get_frameat(filepath: str | Path, time: str, color: bool = True,
                streamnumber: int = 0, ffmpegpath: str | Path = 'ffmpeg',
                loglevel: str = 'quiet') -> NDArray:
    return next(iterread_videofile(filepath, startat=time, nframes=1,
                                   color=color, streamnumber=streamnumber,
                                   ffmpegpath=ffmpegpath, loglevel=loglevel))


@dataclass
class FrameProperties:
    shape: tuple[int, ...]
    size: int
    width: int
    height: int
    pix_fmt: str


def _get_frameproperties(filepath: str | Path, color: bool,
                         streamnumber: int = 0) -> FrameProperties:
    """Convenience function that produces frame characteristics for a given
    video stream. Handy if you want to know the format of a frame that is returned
    by ffmpeg from a pipe."""
    vsi = videofileinfo(filepath)['streams']
    videostreamsinfo = tuple(stream for stream in vsi if stream['codec_type'] == 'video')
    height = videostreamsinfo[streamnumber]['height']
    width = videostreamsinfo[streamnumber]['width']
    if color:
        shape = (height, width, 3)
        size = height * width * 3
        pix_fmt = 'bgr24'
    else:
        shape = (height, width)
        size = height * width
        pix_fmt = 'gray'
    return FrameProperties(shape, size, height, width, pix_fmt)


def _check_loglevelarg(loglevelarg: str) -> None:
    levels = ('quiet', 'panic', 'fatal', 'error', 'warning', 'info',
              'verbose', 'debug', 'trace')
    if loglevelarg not in levels:
        raise ValueError(f"`loglevel` argument ('{loglevelarg}') "
                         f"should be one of: {levels}")


# Audio


def detect_audio_codec(filepath: str | Path, streamnumber: int = 0) -> str | None:
    """Detect the audio codec used in the video file.

    """
    args = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", f"a:{streamnumber}",
        filepath,
    ]
    result = run_ffprobe_json(args)
    streams = result.get("streams", [])
    if not streams:
        return None
    else:
        return streams[0]["codec_name"]


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

    _AUDIO_ENCODER_RE = re.compile(
        rf"^\s*{re.escape(kind)}\S*\s+([^\s]+)",
        re.MULTILINE,
    )

    args = [str(ffmpegpath), "-hide_banner", "-encoders"]
    result = run_ffmpeg(args)
    if result.returncode != 0:
        raise FFmpegError(
            cmd=result.args,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    codecs = {
        match.group(1).lower()
        for match in _AUDIO_ENCODER_RE.finditer(result.stdout)
    }
    codecs.discard("=")
    return codecs


supported_audio_codecs = functools.partial(supported_encoders, "A")

supported_video_codecs = functools.partial(supported_encoders, "V")


def extract_audio(filepath: str | Path, outputpath: str | Path | None = None,
                  overwrite: bool = False, codec: str = 'copy',
                  channel: int | None = None, streamnumber: int = 0,
                  ffmpegpath='ffmpeg',
                  loglevel: str ='quiet') -> Path:
    """Extract audio as wav file.

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
        Channel number to extract. The default None will extract all channels.
    ffmpegpath : str or pathlib.Path, optional
        Path to ffmpeg executable. Default is `ffmpeg`, which means it should
        be in the system path.
    loglevel : {'quiet', 'panic', 'fatal', 'error', 'warning', 'info',
                'verbose', 'debug' ,'trace'}, optional
        Level of info that ffmpeg should print to terminal. Default is
        'quiet'.

    Returns
    -------
    outputpath: Path
        The path of the generated audio file

    """
    filepath = Path(filepath)
    if codec == 'copy':
        codec = detect_audio_codec(str(filepath))
        if not codec:
            raise NoAudioStreamError(f'No audio stream in file "{filepath}"')
    if not codec in supported_audio_codecs(ffmpegpath=ffmpegpath):
        raise ValueError(f'ffmpeg does not support codec "{codec}"')
    ext = AUDIOCODEC_TO_EXTENSION.get(codec)
    if outputpath is None:
        outputpath = Path(filepath).with_suffix(ext)
    else:
        outputpath = Path(outputpath)
    if Path(outputpath).suffix:
        if outputpath.suffix.lower() != ext:
            warnings.warn(f'Specified audio extension ("{outputpath.suffix}") is '
                          f'not the same as the default ("{ext}") for this codec '
                          f'("{codec}"). Proceeding anyway.')
    else:
        outputpath = outputpath.with_suffix(ext)
    if outputpath.exists() and not overwrite:
        raise FileExistsError(f'"{outputpath}" already exists, use `overwrite` parameter')
    _check_loglevelarg(loglevel)
    args = [str(ffmpegpath), '-loglevel' , loglevel, '-y',
            '-i', str(filepath), '-map', f'0:a:{streamnumber}',
            '-vn',
            '-c:a', codec]
    if channel is not None:
        args += ['-af', f'pan=mono|c0=c{channel-1}']
    args += [str(outputpath)]
    result = run_ffmpeg(args)
    return outputpath
