import subprocess
import numpy as np
import json
from pathlib import Path

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
    frames: iterable
        Iterable should produce numpy height x width x channel arrays with
        values ranging from 0 to 255. Frames can be color (3-dim) or gray (
        2-dim).
    filepath: str
        Name of the videofilepath that should be written to.
    framerate: int
        framerate of video in frames per second.
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
    loglevel: str
        Level of info that ffmpeg should print to terminal. Options are 'quiet',
        'panic', 'fatal', 'error', 'warning', 'info', 'verbose', 'debug' ,
        'trace'. Default is quiet.


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
                       ffmpegpath='ffmpeg', loglevel= 'quiet'):
    """
    Parameters
    ----------
    filepath
    startat: str
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
    nframes: int
    color: bool
    ffmpegpath:

    Returns
    -------
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
            if (ar.size == framesize) and ((nframes is None) or (frameno < nframes)):
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


def get_frame(filepath, framenumber, color=True, ffmpegpath='ffmpeg', loglevel= 'quiet'):
    _check_loglevelarg(loglevel)
    frameshape, framesize, frameheight, framewidth, pix_fmt = \
        _get_frameproperties(filepath=filepath, color=color)
    args = [str(ffmpegpath), '-loglevel' , loglevel, '-i', str(filepath)]
    args +=['-vcodec', 'rawvideo',  '-vf', f"select='eq(n\,{framenumber})'",
            '-vframes', '1', '-pix_fmt', pix_fmt,
            '-f', 'rawvideo', 'pipe:1']
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        return np.frombuffer(p.stdout.read(framesize), dtype=np.uint8).reshape(frameshape)

def get_frameat(filepath, time, color=True, ffmpegpath='ffmpeg', loglevel= 'quiet'):
    return next(iterread_videofile(filepath, startat=time, nframes=1, \
                                   color=color, ffmpegpath=ffmpegpath), loglevel=loglevel)

# FIXME do not assume things on audio (i.e. number of channels) and make more versatile
def extract_audio(filepath, outputpath=None, overwrite=False, verbosity=0, ffmpegpath='ffmpeg',
                  loglevel= 'quiet'):
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
            '-codec:a', 'pcm_s24le',
            str(outputpath)]
    with subprocess.Popen(args, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as p:
        out, err = p.communicate()
    if err:
        return err.decode('utf-8')

def _get_frameproperties(filepath, color):
    """Convenience function that produces frame characteristics for a given
    video. Handy if you want to know the format of a frame that is returned
    by ffmpeg from a pipe. """
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