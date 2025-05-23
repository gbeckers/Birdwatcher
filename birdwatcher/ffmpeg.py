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
from pathlib import Path

import numpy as np

from .utils import peek_iterable


__all__ = ['arraytovideo']


class FFmpegError(Exception):
    def __init__(self, cmd, stdout, stderr):
        super().__init__(f'{cmd} error (see stderr output for '
                                    f'detail)')
        self.stdout = stdout
        self.stderr = stderr


def arraytovideo(frames, filepath, framerate, scale=None, crf=17,
                 format='mp4', codec='libx264', pixfmt='yuv420p',
                 ffmpegpath='ffmpeg', loglevel='quiet'):
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
    format : str, default='mp4'
        ffmpeg video format.
    codec : str, default='libx264'
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

    Notes
    -----
    See ffmpeg documentation for more information.

    """
    _check_loglevelarg(loglevel)
    frame, framegen = peek_iterable(frames)
    height, width, *_ = frame.shape
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)   
    if frame.ndim == 2:
        ipixfmt = 'gray'
    else: # frame.ndim == 3:
        ipixfmt = 'bgr24'

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
    if format is not None:
        args += ['-f', f'{format}']
    if crf is not None:
        args += ['-crf', f'{crf}']
    if pixfmt is not None:
        args +=['-pix_fmt', f'{pixfmt}']
    if scale is not None:
        width, height = scale
        args.extend(['-vf', f'scale={width}:{height}'])
    args.extend([filepath, '-y'])
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdout=subprocess.PIPE)
    for frame in framegen:
        # if frame.ndim == 2:
        #     frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        p.stdin.write(frame.astype(np.uint8).tobytes())
    out, err = p.communicate()
    p.stdin.close()
    
    return Path(filepath)


def videofileinfo(filepath, ffprobepath='ffprobe'):
    args = [str(ffprobepath), '-print_format', 'json', '-show_format',
            '-show_streams', str(filepath)]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(p.stdout.read().decode('utf-8'))

## FIXME inform before raising StopIteration that file has no frames
def iterread_videofile(filepath, startat=None, nframes=None, color=True,
                       ffmpegpath='ffmpeg', loglevel='quiet'):
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
    _check_loglevelarg(loglevel)
    frameshape, framesize, frameheight, framewidth, pix_fmt = \
        _get_frameproperties(filepath=filepath, color=color)
    args = [str(ffmpegpath), '-loglevel' , loglevel]
    if startat is not None:
        args.extend(['-ss', startat, '-i', str(filepath)])
    else:
        args.extend(['-i', str(filepath)])
    if nframes is not None:
        args += ['-vframes', str(nframes)]
    args +=['-vcodec', 'rawvideo', '-pix_fmt', pix_fmt,
            '-f', 'rawvideo', 'pipe:1']
    with subprocess.Popen(args, stdout=subprocess.PIPE,
                          stderr=None) as p:
        frameno = 0
        while True:
            data = p.stdout.read(framesize)
            ar = np.frombuffer(data, dtype=np.uint8)
            if (ar.size == framesize) and ((nframes is None) or (frameno <
                                                                 nframes)):
                yield ar.reshape(frameshape)
                frameno += 1
            else:
                break

#TODO check threads code
def count_frames(filepath, threads=8, ffprobepath='ffprobe'):
    args = [str(ffprobepath), '-threads:0', str(threads),
            '-count_frames', '-select_streams', 'v:0', '-show_entries',
            'stream=nb_read_frames', '-print_format', 'json', str(filepath)]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = json.loads(p.stdout.read().decode('utf-8'))
    err = p.stderr.read().decode('utf-8')
    if not out:
        raise FFmpegError(ffprobepath, out, err)
    return int(out['streams'][0]['nb_read_frames'])

# def get_frame_old(filepath, framenumber, color=True, ffmpegpath='ffmpeg'):
#     for frame in iterread_videofile(filepath, startat=None, nframes=framenumber+1,
#                                     color=color, ffmpegpath=ffmpegpath):
#         pass
#     return frame


def get_frame(filepath, framenumber, color=True, ffmpegpath='ffmpeg',
              loglevel= 'quiet'):
    _check_loglevelarg(loglevel)
    frameshape, framesize, frameheight, framewidth, pix_fmt = \
        _get_frameproperties(filepath=filepath, color=color)
    args = [str(ffmpegpath), '-loglevel' , loglevel, '-i', str(filepath)]
    args +=['-vcodec', 'rawvideo',  '-vf', f"select='eq(n\\,{framenumber})'",
            '-vframes', '1', '-pix_fmt', pix_fmt,
            '-f', 'rawvideo', 'pipe:1']
    with subprocess.Popen(args, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as p:
        return np.frombuffer(p.stdout.read(framesize),
                             dtype=np.uint8).reshape(frameshape)


def get_frameat(filepath, time, color=True, ffmpegpath='ffmpeg', 
                loglevel='quiet'):
    return next(iterread_videofile(filepath, startat=time, nframes=1, 
                                   color=color, ffmpegpath=ffmpegpath, 
                                   loglevel=loglevel))


def extract_audio(filepath, outputpath=None, overwrite=False, 
                  codec='pcm_s24le', channel=None, ffmpegpath='ffmpeg', 
                  loglevel='quiet'):
    """Extract audio as wav file.

    Parameters
    ----------
    outputpath : str or pathlib.Path, optional
        Filename and path to write audio to. The default is None, which means 
        the same directoy and name as the video file is used, but then with 
        '.wav' extension.
    overwrite : bool, default=False
        Overwrite if audio file exists or not.
    codec : str, default='pcm_s24le'
        ffmpeg audio codec, with 24-bit pcm as default output.
    channel : int, default=None
        Channel number to extract. The default None will extract all channels.
    ffmpegpath : str or pathlib.Path, optional
        Path to ffmpeg executable. Default is `ffmpeg`, which means it should 
        be in the system path.
    loglevel : {'quiet', 'panic', 'fatal', 'error', 'warning', 'info', 
                'verbose', 'debug' ,'trace'}, optional
        Level of info that ffmpeg should print to terminal. Default is 
        'quiet'.

    """
    filepath = Path(filepath)
    if outputpath is None:
        outputpath = filepath.with_suffix('.wav')
    else:
        outputpath = Path(outputpath)
    if outputpath.exists() and not overwrite:
        raise IOError(f'"{outputpath}" already exists, use overwrite parameter')
    _check_loglevelarg(loglevel)
    args = [str(ffmpegpath), '-loglevel' , loglevel, '-y',
            '-i', str(filepath),
            '-vn',
            '-codec:a', codec]
    if channel is not None:
        args += ['-af', f'pan=mono|c0=c{channel-1}']
    args += [str(outputpath)]
    with subprocess.Popen(args, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as p:
        out, err = p.communicate()
    if err:
        return err.decode('utf-8')


def _get_frameproperties(filepath, color):
    """Convenience function that produces frame characteristics for a given
    video. Handy if you want to know the format of a frame that is returned
    by ffmpeg from a pipe."""
    vfi = videofileinfo(filepath)
    frameheight = vfi['streams'][0]['height']
    framewidth = vfi['streams'][0]['width']
    if color:
        frameshape = (frameheight, framewidth, 3)
        framesize = frameheight * framewidth * 3
        pix_fmt = 'bgr24'
    else:
        frameshape = (frameheight, framewidth)
        framesize = frameheight * framewidth
        pix_fmt = 'gray'
    return frameshape, framesize, frameheight, framewidth, pix_fmt


def _check_loglevelarg(loglevelarg):
    levels = ('quiet', 'panic', 'fatal', 'error', 'warning', 'info',
              'verbose', 'debug', 'trace')
    if loglevelarg not in levels:
        raise ValueError(f"`loglevel` argument ('f{loglevelarg}') "
                         f"should be one of: {levels}")