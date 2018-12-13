import os
from pathlib import Path
import numpy as np
import cv2 as cv
import darr

from .videoinput import VideoFile
from .coordinatearrays import create_coordarray
from .bgsegm import BackgroundSubtractorMOG2, BackgroundSubtractorKNN
from ._version import get_versions

__all__ = ['detect_movementknn', 'batch_detect_movementknn',
           'detect_movementmog2', 'MovementDetector']

class MovementDetector():
    
    _version = get_versions()['version']
    
    def __init__(self, bgsubtractor, learningrate=-1, ignore_firstnframes=0, 
                 focus_rectcoord=None, ignore_rectcoord=None, downscale=None, 
                 morphologyex=2):

        """Detect movement in when iterating over frames, based on a specified
        BackgroundSubtractor object.

        Parameters
        ----------
        bgsubtractor: Birdwatcher BackgroundSubtractor
        learningrate: int
            Default -1.
        morphologyex: int
           Apply morphologyEx function with specifeid size after having
           applied backgroundsubtractor. Default 2.
        focus_rectcoord: len-4 sequence or None
           Only consider pixels within the specified rectangle (h1, h2, w1,
           w2). Default None.
        ignore_firstnframes: int
            Do not consider hits in the first n frames. Default 50.
        ignore_rectcoord: len-4 sequence or None
            Ignore pixels within the specified rectangle (h1, h2, w1, w2).
        downscale: int
            Downscale input image size with specified factor before applying
            algorithm. Default None.
        detect_shadows: bool
            If true, the algorithm will detect shadows and mark them. It
            decreases the speed a bit, so if you do not need this feature,
            set the parameter to false. Default False.

        """

        self.bgsubtractor = bgsubtractor
        self.learningrate = learningrate
        self.ignore_firstnframes = ignore_firstnframes
        self.focus_rectcoord = focus_rectcoord
        self.ignore_rectcoord = ignore_rectcoord
        self.downscale = downscale
        self.morphologyex = morphologyex
        self.mekernel = np.ones((morphologyex, morphologyex), np.uint8)
        self.emptyidx = np.zeros((0,2), dtype=np.uint16)
        self._framesprocessed = 0
        self._emptyframe = None
        
    def apply(self, frame):
        if self.focus_rectcoord is not None:
            if self._framesprocessed == 0:
                self._emptyframe = np.zeros((frame.shape[0], frame.shape[1]),
                                            dtype=np.uint16)
            w1,w2,h1,h2 = self.focus_rectcoord
            frame_rect = frame[h1:h2,w1:w2]
        else:
            frame_rect = frame 
        frame_gray = cv.cvtColor(frame_rect, cv.COLOR_BGR2GRAY)
        if self.ignore_rectcoord is not None:
            iw1,iw2,ih1,ih2 = self.ignore_rectcoord
            frame_gray[ih1:ih2,iw1:iw2] = 0
        if self.downscale is not None:
            frame_gray = cv.resize(frame_gray, None, fx=1/self.downscale,
                                   fy=1/self.downscale,
                                   interpolation=cv.INTER_LINEAR)
        thresh_rect = self.bgsubtractor.apply(image=frame_gray,
                                              learningRate=self.learningrate)
        if self.morphologyex is not None:
            thresh_rect = cv.morphologyEx(thresh_rect, cv.MORPH_OPEN,
                                          self.mekernel)
        if self.downscale is not None:
            thresh_rect = cv.resize(thresh_rect, None, fx=self.downscale,
                                    fy=self.downscale,
                                    interpolation=cv.INTER_LINEAR)
        if self._framesprocessed < self.ignore_firstnframes:
            thresh_rect[:] = 0
        idx = cv.findNonZero(thresh_rect)
        if idx is None:
            idx = self.emptyidx
        else:
            idx = idx[:,0,:]
        if self.focus_rectcoord is not None:
            self._emptyframe[h1:h2,w1:w2] = thresh_rect
            thresh = self._emptyframe
            idx[:,0] += h1
            idx[:,1] += w1
        else:
            thresh = thresh_rect
        self._framesprocessed +=1
        return thresh, idx
    
    def get_params(self):
        return  {'class': str(self.__class__),
                 'learningrate': self.learningrate,
                 'ignore_firstnframes': self.ignore_firstnframes,
                 'focus_rectcoord': self.focus_rectcoord,
                 'ignore_rectcoord': self.ignore_rectcoord,
                 'downscale': self.downscale,
                 'morphologyex': self.morphologyex}


    def _create_bgsubtractor(self):
        "To be implemented by subclass with actual algorithm"

        pass


def batch_detect_movementknn(videofilepaths, nprocesses=6, *args, **kwargs):
    """The reason for having a special batch function, instead of just
    applying functions in a loop, is that compression of coordinate results
    takes a long time and is single-threaded. We therefore do this in
    parallel. Use the `nprocesses` parameter to specify the number of cores
    devoted to this.

    """
    from multiprocessing import Pool

    def f(rar):
        rar.archive(overwrite=True)
        darr.delete_raggedarray(rar)

    tobearchived = []
    for i, videofilepath in enumerate(videofilepaths):
        cd, cc, cm = detect_movementknn(videofilepath, *args, **kwargs)
        tobearchived.append(cd)
        if (len(tobearchived) == nprocesses) or (i == (len(videofilepaths) - 1)):
            with Pool(processes=nprocesses) as pool:
                [i for i in pool.imap_unordered(f, tobearchived)]
            tobearchived = []


def coordcount(coords):
    return np.array([idx.shape[0] for idx in coords.iter_arrays()])

def coordmean(coords):
        return np.array([idx.mean(0) for idx in coords.iter_arrays()])

def _detect_movement(bgsclass, videofilepath, morphologyex=2,
                     analysispath='.', ignore_rectcoord=None,
                     ignore_firstnframes=50, **kwargs):

    vf = VideoFile(videofilepath)
    bgs = bgsclass(**kwargs)
    algostr = str(bgs)
    md = MovementDetector(bgs, ignore_firstnframes=ignore_firstnframes,
                          ignore_rectcoord=ignore_rectcoord)
    analysispath = Path(analysispath) / Path(
        f'{vf.filepath.stem}_movement_{algostr}_me{ morphologyex}')
    if not analysispath.exists():
        os.mkdir(analysispath)

    metadata = bgs.get_params()
    cd = create_coordarray(analysispath / 'coordinates.drarr',
                           videofile=vf, metadata=metadata, overwrite=True)
    with cd._view():
        for i, frame in enumerate(vf.iter_frames()):
            thresh, idx = md.apply(frame)
            cd.append(idx)
    cc = darr.asarray(analysispath / 'coordscount.darr', coordcount(cd),
                      metadata=metadata, overwrite=True)
    cm = darr.asarray(analysispath / 'coordsmean.darr', coordmean(cd),
                      metadata=metadata, overwrite=True)
    return cd, cc, cm


def detect_movementknn(videofilepath, morphologyex=2, analysispath='.',
                       ignore_rectcoord=None, ignore_firstnframes=50,
                       **kwargs):
    cd, cc, cm = _detect_movement(bgsclass=BackgroundSubtractorKNN, \
                                  videofilepath=videofilepath,
                                  morphologyex=morphologyex,
                                  analysispath=analysispath,
                                  ignore_rectcoord=ignore_rectcoord,
                                  ignore_firstnframes=ignore_firstnframes,
                                  **kwargs)
    return cd, cc, cm


def detect_movementmog2(videofilepath, morphologyex=2, analysispath='.',
                       ignore_rectcoord=None, ignore_firstnframes=50,
                       **kwargs):
    cd, cc, cm = _detect_movement(bgsclass=BackgroundSubtractorMOG2, \
                                  videofilepath=videofilepath,
                                  morphologyex=morphologyex,
                                  analysispath=analysispath,
                                  ignore_rectcoord=ignore_rectcoord,
                                  ignore_firstnframes=ignore_firstnframes,
                                  **kwargs)
    return cd, cc, cm


def calc_meanframe(videofilepath):
    vf = VideoFile(videofilepath)
    meanframe = vf.get_framebynumber(0).astype('float64')
    meanframe[:] = 0.0
    for i, frame in enumerate(vf.iter_frames()):
        meanframe += frame
    meanframe /= i
    return meanframe