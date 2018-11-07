import sys
from contextlib import contextmanager
import pathlib
import cv2 as cv
from darr import create_raggedarray
from .coordinatedata import CoordinateData

__all__ = ['VideoFile']


class VideoFile():
    
    _version = '0.1.0'
    
    def __init__(self, filepath):
        self.filepath = fp = pathlib.Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f'"{fp.name}" does not exist')
        vp = self._get_videoproperties()
        self.fourcccode = vp['fourcc']
        self.fourcc = cv.VideoWriter_fourcc(*self.fourcccode)
        self.shape = vp['shape']
        self.framerate = vp['framerate']
        self.duration = vp['duration']
        self.format = vp['format']
        self.framecount = vp['framecount']
    
    def _get_videoproperties(self):
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
        d['version'] = self._version
        cap.release()
        return d
    
    def derive_filepath(self, s, suffix=None, path=None):
        parts = list(self.filepath.parts)
        stem = self.filepath.stem
        if suffix is None:
            suffix = self.filepath.suffix
        filename = f'{stem}_{s}{suffix}'
        if path is None:
            dpath =  pathlib.Path('/'.join(parts[:-1] + [filename]))
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
        if stopframe is None:
            stopframe = self.framecount
        for i in range(stopframe):
            #cap.isOpened()
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                cap.release()
                break
        cap.release()

    def create_coordarray(self, s, path=None, metadata=None, overwrite=False):
        fpath = self.derive_filepath(s, suffix='.drarr', path=path)
        if metadata is None:
            metadata = {}
        for key, item in (self._get_videoproperties().items()):
            metadata[f'videofile_{key}'] = item
        ra = create_raggedarray(fpath, atom=(2,), dtype='uint16',
                                metadata=metadata, accessmode='r+',
                                overwrite=overwrite)
        return CoordinateData(ra.path, accessmode='r+')

    @contextmanager
    def open_videowriter(self, s, path=None):
        video_writer = self.derive_videowriter(s, path=path)
        yield video_writer
        video_writer.release()