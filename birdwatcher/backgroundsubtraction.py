"""This module provides background subtractors. Background subtraction is a
major preprocessing step in many computer vision applications. It extracts the
moving foreground from static background.

All classes are based on OpenCV's background subtraction algorithms. They
can be used for example in movement detection. Note that OpenCV's API to
parameters of these algorithms is inconsistent. Sometimes parameters can be
provided at instantiation, sometimes you can change them with a method. In
Birdwatcher you can only provide parameters to background subtractor objects
at instantiation. The parameter names follow that of OpenCV. The parameter
names follow those of OpenCV to avoide confusion.

"""

import cv2 as cv


__all__ = ['BackgroundSubtractorMOG2', 'BackgroundSubtractorKNN',
           'BackgroundSubtractorLSBP']


class BaseBackgroundSubtractor:

    _initparams = {}                # to be implemented by subclass
    _setparams = {}                 # to be implemented by subclass
    _bgsubtractorcreatefunc = None  # to be implemented by subclass

    def __init__(self, **kwargs):
        # create background subtractor
        initkeys = set(kwargs.keys()) & set(self._initparams.keys())
        initparams = self._initparams.copy()
        for initkey in initkeys:
            initparams[initkey] = kwargs[initkey]
        self._bgs = self._bgsubtractorcreatefunc(**initparams)
        setkeys = set(kwargs.keys()) & set(self._setparams.keys())
        setparams = self._setparams.copy()
        for setkey in setkeys:
            setparams[setkey] = kwargs[setkey]
        self._set_params(**setparams)
        self._params = {**initparams, **setparams}

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"<{self.__class__.__name__} {hex(id(self))}>"

    def _set_params(self, **kwargs):
        for key, val in kwargs.items():
            methodname = f'set{key}'
            self._bgs.__getattribute__(methodname)(val)

    def get_params(self):
        return self._params

    def apply(self, image, fgmask=None, learningRate=-1):
        return self._bgs.apply(image=image, fgmask=fgmask,
                               learningRate=learningRate)


class BackgroundSubtractorKNN(BaseBackgroundSubtractor):

    """Wraps OpenCV's `BackgroundSubtractorKNN` class.

    Parameters
    ----------
    History: int
        Length of the history. Default 50.
    kNNSamples: int
        The number of neighbours, the k in the kNN. K is the number of
        samples that need to be within dist2Threshold in order to decide
        that that pixel is matching the kNN background model. Default 10.
    NSamples: int
        The number of data samples in the background model. Default 6.
    Dist2Threshold: float
        Threshold on the squared distance between the pixel and the sample
        to decide whether a pixel is close to that sample. This parameter
        does not affect the background update. Default 500.

    """

    _setparams = {'History': 5,
                  'kNNSamples': 10,
                  'NSamples': 6,
                  'Dist2Threshold': 500}

    _bgsubtractorcreatefunc = cv.createBackgroundSubtractorKNN


class BackgroundSubtractorMOG2(BaseBackgroundSubtractor):

    _setparams = {'History': 5,
                  'ComplexityReductionThreshold': 0.05,
                  'BackgroundRatio': 0.1,
                  'NMixtures': 7,
                  'VarInit': 15,
                  'VarMin': 4,
                  'VarMax': 75,
                  'VarThreshold': 10,
                  'VarThresholdGen': 9,
                  'ShadowThreshold': 0.5,
                  'ShadowValue': 127}

    _bgsubtractorcreatefunc = cv.createBackgroundSubtractorMOG2


class BackgroundSubtractorLSBP(BaseBackgroundSubtractor):

    _initparams = {'mc': 0,
                   'nSamples': 20,
                   'LSBPRadius': 16,
                   'Tlower': 2.0,
                   'Tupper': 32.0,
                   'Tinc': 1.0,
                   'Tdec': 0.05,
                   'Rscale': 10.0,
                   'Rincdec': 0.005,
                   'noiseRemovalThresholdFacBG': 0.0004,
                   'noiseRemovalThresholdFacFG': 0.0008,
                   'LSBPthreshold': 8,
                   'minCount': 2}

    _bgsubtractorcreatefunc = cv.bgsegm.createBackgroundSubtractorLSBP

