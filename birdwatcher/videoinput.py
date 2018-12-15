"""This module contains classes and functions to work with video files. For
now this depends on OpenCV's VideoCapture class.

"""

import sys
import pathlib
from contextlib import contextmanager
import cv2 as cv

__all__ = ['VideoFile', 'testvideosmall']


class VideoFile():
    """Read video files.

    Parameters
    ----------
    filepath: str of pathlib.Path
        Path to videofile.

    """

    def __init__(self, filepath):

        self.filepath = fp = pathlib.Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f'"{filepath}" does not exist')
        vp = self.get_properties()
        self.fourcccode = vp['fourcc']
        self._fourcc = cv.VideoWriter_fourcc(*self.fourcccode)
        self._shape = vp['shape']
        self._framerate = vp['framerate']
        self._duration = vp['duration']
        self._format = vp['format']
        self._framecount = vp['framecount']
        self._nframes = None

    @property
    def duration(self):
        """Duration of video in secomds."""
        return self._duration

    @property
    def format(self):
        """Video format."""
        return self._format

    @property
    def fourcc(self):
        """Four character code video codec."""
        return self._fourcc

    @property
    def framecount(self):
        """Number of frame video in video as reported by header. This is not
        necessarily accurate. If accuracy is important use `nframes`."""
        return self._framerate

    @property
    def framerate(self):
        """Frame rate of video in frames / second."""
        return self._framerate

    @property
    def nframes(self):
        """Number of frames in video. This is determined by reading the
        whole video stream, which may take a long time depending on how large
        it is. This is accurate though, as opposed to the `framecount`
        attribute which is not always reliable, but fast.

        """
        if self._nframes is not None:
            return self._nframes
        else:
            for i,_ in enumerate(self.iter_frames(),1):
                pass
            self._nframes = i
            return i
    @property
    def shape(self):
        """Shape (width, height) of video frame."""
        return self._shape

    def get_properties(self, affix=None):
        d = {}
        cap = cv.VideoCapture(str(self.filepath))
        d['width'] = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        d['height'] = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        d['shape'] = (d['width'], d['height'])
        d['framecount'] = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        d['framerate'] = cap.get(cv.CAP_PROP_FPS)
        d['fourcc'] = int(cap.get(cv.CAP_PROP_FOURCC)).to_bytes(4, byteorder=sys.byteorder).decode('utf8')
        d['format'] = cap.get(cv.CAP_PROP_FORMAT)
        d['duration'] = d['framecount'] / d['framerate']
        d['filename'] = self.filepath.parts[-1]
        cap.release()
        if affix is not None:
            ad = {}
            for key, item in (d.items()):
                ad[f'{affix}{key}'] = item
            d = ad
        return d
    
    def derive_filepath(self, s, suffix=None, path=None):
        stem = self.filepath.stem
        if suffix is None:
            suffix = self.filepath.suffix
        filename = f'{stem}_{s}{suffix}'
        if path is None:
            dpath =  self.filepath.parent / filename
        else:
            dpath = pathlib.Path(path) / filename
        return dpath
    
    def derive_videowriter(self, s, path=None):
        trackfn = str(self.derive_filepath(s, suffix='.avi', path=path))
        return cv.VideoWriter(trackfn, cv.VideoWriter_fourcc(*'MJPG'),
                              self._framerate, self.shape, True)
    def get_framebynumber(self, framenumber):
        cap = cv.VideoCapture(str(self.filepath)) 
        res = cap.set(cv.CAP_PROP_POS_FRAMES, framenumber)
        if res:
            ret, frame = cap.read()
        else:
            raise ValueError(f'frame number {framenumber} could not be read')
        cap.release()
        return frame
        
    def iter_frames(self, stopframe=None):
        cap = cv.VideoCapture(str(self.filepath))
        frameno = 0
        while(True):
            ret, frame = cap.read()
            if ret and ((stopframe is None) or (frameno < stopframe)):
                yield frame
                frameno +=1
            else:
                cap.release()
                break
        cap.release()

    @contextmanager
    def open_videowriter(self, s, path=None):
        video_writer = self.derive_videowriter(s, path=path)
        yield video_writer
        video_writer.release()


def testvideosmall():
    """A 20-s video of a zebra finch for testing purposes.

    Returns
    -------
    VideoFile
        An instance of Birdwatcher's VideoFile class.

    """
    file = 'zf20s_low.mp4'
    path = pathlib.Path(__file__).parent / 'testvideos' / file
    return VideoFile(path)