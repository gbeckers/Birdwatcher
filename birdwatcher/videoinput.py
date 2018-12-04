import sys
import pathlib
from contextlib import contextmanager
import cv2 as cv

__all__ = ['VideoFile', 'testvideosmall']


class VideoFile():

    
    def __init__(self, filepath):
        self.filepath = fp = pathlib.Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f'"{filepath}" does not exist')
        vp = self.get_properties()
        self.fourcccode = vp['fourcc']
        self.fourcc = cv.VideoWriter_fourcc(*self.fourcccode)
        self.shape = vp['shape']
        self.framerate = vp['framerate']
        self.duration = vp['duration']
        self.format = vp['format']
        self.framecount = vp['framecount']
    
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
                            self.framerate, self.shape, True)
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


def testvideosmall(file='zf20s_low.mp4'):
    path = pathlib.Path(__file__).parent / 'testvideos' / file
    return VideoFile(path)