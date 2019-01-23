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


def arraytovideo(frames, filename, framerate, crf=17, format='mp4',
                 codec='libx264', pixfmt='yuv420p', ffmpegpath='ffmpeg'):
    """Writes an iterable of numpy frames as video file using ffmpeg.

    Parameters
    ----------
    frames: iterable
        Iterable should produce numpy height x width x channel arrays with
        values ranging from 0 to 255. Frames can be color (3-dim) or gray (
        2-dim)
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
    frame, framegen = peek_iterable(frames)
    height, width, *_ = frame.shape
    filename = str(filename)
    if frame.ndim == 2:
        ipixfmt = 'gray'
    elif frame.ndim == 3:
        ipixfmt = 'bgr24'
    args = [str(ffmpegpath),
            #'-hwaccel',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-pix_fmt', ipixfmt,
            '-r', f'{framerate}',
            '-s', f'{width}x{height}', '-i', 'pipe:',
            '-vcodec', f'{codec}',
            '-f', f'{format}', '-crf', f'{crf}',
            '-pix_fmt', f'{pixfmt}',
            filename, '-y']
    print(args)
    p = subprocess.Popen(args, stdin=subprocess.PIPE)
    for frame in framegen:
        # if frame.ndim == 2:
        #     frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        p.stdin.write(frame.astype(np.uint8).tobytes())
    p.stdin.close()
    out, err = p.communicate()
    return Path(filename)


def videofileinfo(filepath, ffprobepath='ffprobe'):
    args = [str(ffprobepath), '-print_format', 'json', '-show_format',
            '-show_streams', str(filepath)]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(p.stdout.read().decode('utf-8'))


def iterread_videofile(filepath, startat=None, nframes=None, color=True,
                       ffmpegpath='ffmpeg'):
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
    args = [str(ffmpegpath), '-i', str(filepath)]
    if startat is not None:
        args += ['-ss', startat]
    if nframes is not None:
        args += ['-vframes', str(nframes)]
    args +=['-vcodec', 'rawvideo', '-pix_fmt', pix_fmt,
            '-f', 'rawvideo', 'pipe:1']
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=None) as p:
        frameno = 0
        while True:
            data = p.stdout.read(framesize)
            ar = np.frombuffer(data, dtype=np.uint8)
            if (ar.size == framesize) and ((nframes is None) or (frameno < nframes)):
                yield ar.reshape(frameshape)
                frameno += 1
            else:
                break

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


def get_frameat(filepath, time, color=True,ffmpegpath='ffmpeg'):
    return next(iterread_videofile(filepath, startat=time, nframes=1, \
                                   color=color, ffmpegpath=ffmpegpath))