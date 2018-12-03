import os
from pathlib import Path
import numpy as np
import cv2 as cv
import darr

from .videoinput import VideoFile
from. coordinatearrays import create_coordarray
from ._version import get_versions

__all__ = ['detect_movementknn', 'batch_detect_movementknn',
           'detect_movementmog2']

class DetectMovement():
    
    _version = get_versions()['version']
    
    def __init__(self, bgsubtractor, learningrate=-1, ignore_firstnframes=0, 
                 focus_rectcoord=None, ignore_rectcoord=None, downscale=None, 
                 morphologyex=None, bgsubtractorparams=None):
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
        self.bgsubtractorparams = bgsubtractorparams
        self._emptyframe = None
        
    def apply(self, frame):
        if self.focus_rectcoord is not None:
            if self._framesprocessed == 0:
                self._emptyframe = np.zeros((frame.shape[0], frame.shape[0]),
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
        params = {'detector_class': str(self.__class__),
                  'detector_classversion': self._version,
                  'detector_learningrate': self.learningrate,
                  'detector_ignore_firstnframes': self.ignore_firstnframes,
                  'detector_focus_rectcoord': self.focus_rectcoord,
                  'detector_ignore_rectcoord': self.ignore_rectcoord,
                  'detector_downscale': self.downscale,
                  'detector_morphologyex': self.morphologyex}
        if self.bgsubtractorparams is not None:
            params.update(self.bgsubtractorparams)
        return params

    def _create_bgsubtractor(self):
        "To be implemented by subclass with actual algorithm"

        pass


class DetectMovementKnn(DetectMovement):
    
    _version = get_versions()['version']
    
    def __init__(self, history=1, knnsamples=10, nsamples=6, dist2threshold=500, 
                 learningrate=-1, morphologyex=2, focus_rectcoord=None,
                 ignore_firstnframes=50, ignore_rectcoord=None,
                 downscale=None, detect_shadows=False):
        self.history = history
        self.knnsamples = knnsamples
        self.nsamples = nsamples
        self.dist2threshold = dist2threshold
        self.detect_shadows = detect_shadows
        self._knnbgsubtractor = self._create_bgsubtractor()
        bgsubtractorparams = self._get_bgsubtractorparams()
        
        super().__init__(bgsubtractor=self._knnbgsubtractor,
                         learningrate=learningrate,
                         morphologyex=morphologyex,
                         ignore_firstnframes=ignore_firstnframes,
                         focus_rectcoord=focus_rectcoord, 
                         ignore_rectcoord=ignore_rectcoord, 
                         downscale=downscale,
                         bgsubtractorparams=bgsubtractorparams)
    
    def _create_bgsubtractor(self):
        bgsubtractor = cv.createBackgroundSubtractorKNN(detectShadows=self.detect_shadows)
        bgsubtractor.setHistory(self.history)
        bgsubtractor.setkNNSamples(self.knnsamples)
        bgsubtractor.setNSamples(self.nsamples)
        bgsubtractor.setDist2Threshold(self.dist2threshold)
        return bgsubtractor
    
    def _get_bgsubtractorparams(self):
        bgs = self._knnbgsubtractor
        return {'knn_classversion': self._version,
                'knn_history': bgs.getHistory(),
                'knn_knnsamples': bgs.getkNNSamples(),
                'knn_nsamples': bgs.getNSamples(),
                'knn_dist2threshold': bgs.getDist2Threshold(),
                'knn_detect_shadows': self.detect_shadows}


class DetectMovementMOG2(DetectMovement):

    _version = get_versions()['version']

    def __init__(self, history=5, complexityreductionrhreshold=0.05,
                 backgroundratio=0.1, nmmixtures=7, learningrate=-1,
                 morphologyex=2, focus_rectcoord=None,
                 ignore_firstnframes=50, ignore_rectcoord=None,
                 downscale=None, detect_shadows=False):

        self.history = history
        self.complexityreductionrhreshold = complexityreductionrhreshold
        self.backgroundratio = backgroundratio
        self.nmmixtures = nmmixtures
        self.detect_shadows = detect_shadows
        self._mog2bgsubtractor = self._create_bgsubtractor()
        bgsubtractorparams = self._get_bgsubtractorparams()

        super().__init__(bgsubtractor=self._mog2bgsubtractor,
                         learningrate=learningrate,
                         morphologyex=morphologyex,
                         ignore_firstnframes=ignore_firstnframes,
                         focus_rectcoord=focus_rectcoord,
                         ignore_rectcoord=ignore_rectcoord,
                         downscale=downscale,
                         bgsubtractorparams=bgsubtractorparams)

    def _create_bgsubtractor(self):
        bgsubtractor = cv.createBackgroundSubtractorMOG2(detectShadows=self.detect_shadows)
        bgsubtractor.setHistory(self.history)
        bgsubtractor.setComplexityReductionThreshold(
            self.complexityreductionrhreshold)
        bgsubtractor.setBackgroundRatio(self.backgroundratio)
        bgsubtractor.setNMixtures(self.nmmixtures)
        return bgsubtractor


    def _get_bgsubtractorparams(self):
        bgs = self._mog2bgsubtractor
        return {'mog2_classversion': self._version,
                'mog2_history': bgs.getHistory(),
                'mog2_complexityreductionrhreshold':
                    bgs.getComplexityReductionThreshold(),
                'mog2_backgroundratio': bgs.getBackgroundRatio(),
                'mog2_nmixtues': bgs.getNMixtures(),
                'mog2_detect_shadows': self.detect_shadows}


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

def _detect_movement(algorithmclass, videofilepath, morphologyex=2,
                     analysispath='.', ignore_rectcoord=None,
                     ignore_firstnframes=50, **kwargs):

    vf = VideoFile(videofilepath)
    analysispath = Path(analysispath) / Path(
        f'{vf.filepath.stem}_movement_{algorithmclass}_me{morphologyex}')
    if not analysispath.exists():
        os.mkdir(analysispath)
    dm = algorithmclass(morphologyex=morphologyex,
                        ignore_rectcoord=ignore_rectcoord,
                        ignore_firstnframes=ignore_firstnframes, **kwargs)
    metadata = dm.get_params()
    cd = create_coordarray(analysispath / 'coordinates.drarr',
                           videofile=vf, metadata=metadata, overwrite=True)
    with cd._view():
        for i, frame in enumerate(vf.iter_frames()):
            thresh, idx = dm.apply(frame)
            cd.append(idx)
    cc = darr.asarray(analysispath / 'coordscount.darr', coordcount(cd),
                      metadata=metadata, overwrite=True)
    cm = darr.asarray(analysispath / 'coordsmean.darr', coordmean(cd),
                      metadata=metadata, overwrite=True)
    return cd, cc, cm


def detect_movementknn(videofilepath, morphologyex=2, analysispath='.',
                       ignore_rectcoord=None, ignore_firstnframes=50,
                       **kwargs):
    cd, cc, cm = _detect_movement(algorithmclass=DetectMovementKnn,\
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
    cd, cc, cm = _detect_movement(algorithmclass=DetectMovementMOG2,\
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