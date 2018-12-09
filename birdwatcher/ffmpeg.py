import subprocess
import numpy as np
from pathlib import Path
from .utils import peek_iterable

__all__ = ['arraytovideo']


def arraytovideo(frames, filename, framerate, crf=17, format='mp4',
                 codec='libx264', pixfmt='yuv420p', ffmpegpath='ffmpeg'):
    """Writes an iterable of numpy frames as video file using ffmpeg.

    Parameters
    ----------
    frames: iterable
        Iterable should produce numpy height x width x channel arrays with
        values ranging from 0 to 255.
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
    args = [str(ffmpegpath), '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-r', f'{framerate}',
            '-s', f'{width}x{height}', '-i', 'pipe:',
            '-vcodec', f'{codec}',
            '-f', f'{format}', '-crf', f'{crf}',
            '-pix_fmt', f'{pixfmt}',
            filename, '-y']
    # print(args)
    p = subprocess.Popen(args, stdin=subprocess.PIPE)
    for frame in framegen:
        p.stdin.write(frame.astype(np.uint8).tobytes())
    p.terminate()
    return Path(filename)
