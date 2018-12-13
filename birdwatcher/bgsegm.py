
import cv2 as cv


__all__ = ['BackgroundSubtractorMOG2', 'BackgroundSubtractorKNN',
           'BackgroundSubtractorLSBP']

class BackgroundSubtractor:

    _initparams = {}  # to be implemented by subclass
    _setparams = {} # to be implemented by subclass
    _bgsubtractorcreatefunc = None # to be implemented by subclass

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

    def _set_params(self, **kwargs):
        for key, val in kwargs.items():
            methodname = f'set{key}'
            self._bgs.__getattribute__(methodname)(val)

    def get_params(self):
        return self._params

    def apply(self, image, fgmask=None, learningRate=-1):
        return self._bgs.apply(image=image, fgmask=fgmask,
                               learningRate=learningRate)




class BackgroundSubtractorKNN(BackgroundSubtractor):

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
                   'Dist2Threshold': 500,}

    _bgsubtractorcreatefunc = cv.createBackgroundSubtractorKNN


class BackgroundSubtractorMOG2(BackgroundSubtractor):

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
                  'ShadowValue': 127
                  }

    _bgsubtractorcreatefunc = cv.createBackgroundSubtractorMOG2


class BackgroundSubtractorLSBP(BackgroundSubtractor):

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
                   'minCount': 2
                   }

    _bgsubtractorcreatefunc = cv.bgsegm.createBackgroundSubtractorLSBP

