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
        """Get parameters of the background subtraction algorithm.

        Returns
        -------
        dict
            Dictionary with algorithm parameters.

        """
        return self._params

    def apply(self, image, fgmask=None, learningRate=-1.0):
        """Computes a foreground mask.

        Parameters
        ----------
        image: numpy array image
            Next video frame.
        fgmask: numpy array image
            The output foreground mask as an 8-bit binary image.
        learningRate: float
            The value between 0 and 1 that indicates how fast the background
            model is learnt. Negative parameter value makes the algorithm to
            use some automatically chosen learning rate. 0 means that the
            background model is not updated at all, 1 means that the background
            model is completely reinitialized from the last frame.

        Returns
        -------
        numpy array image
            The output foreground mask as an 8-bit binary image.

        """
        return self._bgs.apply(image=image, fgmask=fgmask,
                               learningRate=learningRate)


class BackgroundSubtractorKNN(BaseBackgroundSubtractor):

    """Wraps OpenCV's `BackgroundSubtractorKNN` class. Parameter names follow
    those in OpenCV.

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
    DetectShadows: bool
        If true, the algorithm detects shadows and marks them. Default False.
    ShadowThreshold: float
        A shadow is detected if pixel is a darker version of the background.
        The shadow threshold is a threshold defining how much darker the
        shadow can be. 0.5 means that if a pixel is more than twice darker
        then it is not shadow. Deault 0.5.
    ShadowValue: int
        Shadow value is the value used to mark shadows in the foreground mask.
        Value 0 in the mask always means background, 255 means foreground.
        Default value is 127.

    """

    _initparams = {}
    _setparams = {'History': 5,
                  'kNNSamples': 10,
                  'NSamples': 6,
                  'Dist2Threshold': 500,
                  'DetectShadows': False,
                  'ShadowsThreshold': 0.5,
                  'ShadowValue': 127}

    _bgsubtractorcreatefunc = cv.createBackgroundSubtractorKNN


class BackgroundSubtractorMOG2(BaseBackgroundSubtractor):
    """Wraps OpenCV's `BackgroundSubtractorMOG2` class. Parameter names follow
    those in OpenCV.

    Parameters
    ----------
    History: int
        Length of the history. Default 50.
    ComplexityReductionThreshold: float
        This parameter defines the number of samples needed to accept to prove
        the component exists. CT=0.05 is a default value for all the samples.
        By setting CT=0 you get an algorithm very similar to the standard
        Stauffer&Grimson algorithm.
    BackgroundRatio: float
        If a foreground pixel keeps semi-constant value for about
        backgroundRatio*history frames, it's considered background and added to
        the model as a center of a new component. It corresponds to TB
        parameter in the paper. Default 0.1.
    NMixtures: int
        The number of gaussian components in the background model. Default 7.
    VarInit: int
        The initial variance of each gaussian component. Default 15.
    VarMin: int
        The minimum variance of each gaussian component. Default 4.
    VarMax: int
        The maximum variance of each gaussian component. Default 75.
    VarThreshold: int
        The variance threshold for the pixel-model match. The main threshold on
        the squared Mahalanobis distance to decide if the sample is well
        described by the background model or not. Related to Cthr from the
        paper.
    VarThresholdGen: int
        The variance threshold for the pixel-model match used for new mixture
        component generation. Threshold for the squared Mahalanobis distance
        that helps decide when a sample is close to the existing components
        (corresponds to Tg in the paper). If a pixel is not close to any
        component, it is considered foreground or added as a new component.
        3 sigma => Tg=3*3=9 is default. A smaller Tg value generates more
        components. A higher Tg value may result in a small number of
        components but they can grow too large.
    ShadowThreshold: float
        A shadow is detected if pixel is a darker version of the background.
        The shadow threshold is a threshold defining how much darker the
        shadow can be. 0.5 means that if a pixel is more than twice darker
        then it is not shadow. Deault 0.5.
    ShadowValue: int
        Shadow value is the value used to mark shadows in the foreground mask.
        Value 0 in the mask always means background, 255 means foreground.
        Default value is 127.

    """

    _setparams = {'History': 5,
                  'ComplexityReductionThreshold': 0.05,
                  'BackgroundRatio': 0.1,
                  'NMixtures': 7,
                  'VarInit': 15,
                  'VarMin': 4,
                  'VarMax': 75,
                  'VarThreshold': 10,
                  'VarThresholdGen': 9,
                  'DetectShadows': False,
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

