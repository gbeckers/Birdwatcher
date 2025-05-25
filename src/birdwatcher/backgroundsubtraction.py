"""This module provides background subtractors. Background subtraction is a
major preprocessing step in many computer vision applications. It extracts the
moving foreground from static background.

All classes are based on OpenCV's background subtraction algorithms. They
can be used for example in movement detection. Note that OpenCV's API to
parameters of these algorithms is inconsistent. Sometimes parameters can be
provided at instantiation, sometimes you can change them with a method. In
Birdwatcher you can only provide parameters to background subtractor objects
at instantiation. The parameter names follow those of OpenCV to avoid
confusion.

"""
# TODO: incorporate the use of https://github.com/andrewssobral/bgslibrary

import cv2 as cv
import numpy as np


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

    def apply(self, frame, fgmask=None, learningRate=-1.0):
        """Computes a foreground mask.

        Parameters
        ----------
        frame : numpy array image
            Next video frame.
        fgmask : numpy array image, optional
            The output foreground mask as an 8-bit binary image.
        learningRate : float, optional
            The value between 0 and 1 that indicates how fast the background
            model is learnt. The default negative parameter value (-1.0) makes
            the algorithm to use some automatically chosen learning rate. 0
            means that the background model is not updated at all, 1 means
            that the background model is completely reinitialized from the
            last frame.

        Returns
        -------
        image frame
            The output foreground mask as an 8-bit image.

        """

        return self._bgs.apply(image=frame, fgmask=fgmask,
                               learningRate=learningRate)


class BackgroundSubtractorKNN(BaseBackgroundSubtractor):
    """Wraps OpenCV's `BackgroundSubtractorKNN` class. Parameter names follow
    those in OpenCV.

    Parameters
    ----------
    History : int, default=5
        Length of the history.
    kNNSamples : int, default=10
        The number of neighbours, the k in the kNN. K is the number of
        samples that need to be within dist2Threshold in order to decide
        that that pixel is matching the kNN background model.
    NSamples : int, default=6
        The number of data samples in the background model.
    Dist2Threshold : float, default=500
        Threshold on the squared distance between the pixel and the sample
        to decide whether a pixel is close to that sample. This parameter
        does not affect the background update.
    DetectShadows : bool, default=False
        If true, the algorithm detects shadows and marks them.
    ShadowThreshold : float, default=0.5
        A shadow is detected if pixel is a darker version of the background.
        The shadow threshold is a threshold defining how much darker the
        shadow can be. 0.5 means that if a pixel is more than twice darker
        then it is not shadow.
    ShadowValue : int, default=127
        Shadow value is the value used to mark shadows in the foreground mask.
        Value 0 in the mask always means background, 255 means foreground.

    """
    
    _setparams = {'History': 5,
                  'kNNSamples': 10,
                  'NSamples': 6,
                  'Dist2Threshold': 500,
                  'DetectShadows': False,
                  'ShadowThreshold': 0.5,
                  'ShadowValue': 127}

    _bgsubtractorcreatefunc = cv.createBackgroundSubtractorKNN


class BackgroundSubtractorMOG2(BaseBackgroundSubtractor):
    """Wraps OpenCV's `BackgroundSubtractorMOG2` class. Parameter names follow
    those in OpenCV.

    Parameters
    ----------
    History : int, default=3
        Length of the history.
    ComplexityReductionThreshold : float, default=0.5
        This parameter defines the number of samples needed to accept to prove
        the component exists. CT=0.05 is a default value for all the samples.
        By setting CT=0 you get an algorithm very similar to the standard
        Stauffer&Grimson algorithm.
    BackgroundRatio : float, default=0.1
        If a foreground pixel keeps semi-constant value for about
        backgroundRatio*history frames, it's considered background and added
        to the model as a center of a new component. It corresponds to TB
        parameter in the paper.
    NMixtures : int, default=7
        The number of gaussian components in the background model.
    VarInit : int, default=15
        The initial variance of each gaussian component.
    VarMin : int, default=10
        The minimum variance of each gaussian component.
    VarMax : int, default=75
        The maximum variance of each gaussian component.
    VarThreshold : int, default=70
        The variance threshold for the pixel-model match. The main threshold
        on the squared Mahalanobis distance to decide if the sample is well
        described by the background model or not. Related to Cthr from the
        paper.
    VarThresholdGen : int, default=9
        The variance threshold for the pixel-model match used for new mixture
        component generation. Threshold for the squared Mahalanobis distance
        that helps decide when a sample is close to the existing components
        (corresponds to Tg in the paper). If a pixel is not close to any
        component, it is considered foreground or added as a new component.
        3 sigma => Tg=3*3=9 is default. A smaller Tg value generates more
        components. A higher Tg value may result in a small number of
        components but they can grow too large.
    DetectShadows : bool, default=False
        If true, the algorithm detects shadows and marks them.
    ShadowThreshold : float, default=0.5
        A shadow is detected if pixel is a darker version of the background.
        The shadow threshold is a threshold defining how much darker the
        shadow can be. 0.5 means that if a pixel is more than twice darker
        then it is not shadow.
    ShadowValue : int, default=0
        Shadow value is the value used to mark shadows in the foreground mask.
        Value 0 in the mask always means background, 255 means foreground.

    """
    
    _setparams = {'History': 3,
                  'ComplexityReductionThreshold': 0.05,
                  'BackgroundRatio': 0.1,
                  'NMixtures': 7,
                  'VarInit': 15,
                  'VarMin': 10,
                  'VarMax': 75,
                  'VarThreshold': 70,
                  'VarThresholdGen': 9,
                  'DetectShadows': False,
                  'ShadowThreshold': 0.5,
                  'ShadowValue': 0}

    _bgsubtractorcreatefunc = cv.createBackgroundSubtractorMOG2

#TODO this bgs does nor work with roi or nroi
class BackgroundSubtractorLSBP(BaseBackgroundSubtractor):
    """Wraps OpenCV's `BackgroundSubtractorLSBP` class. Parameter names follow
    those in OpenCV.

    Parameters
    ----------
    mc : int, default=0
        Whether to use camera motion compensation.
    nSamples : int, default=20
        Number of samples to maintain at each point of the frame.
    LSBPRadius : int, default=16
        LSBP descriptor radius.
    Tlower : float, default=2.0
        Lower bound for T-values. See [103] for details.
    Tupper : float, default=32.0
        Upper bound for T-values. See [103] for details.
    Tinc : float, default=1.0
        Increase step for T-values.
    Tdec : float, default=0.05
        Decrease step for T-values.
    Rscale : float, default=10.0
        Scale coefficient for threshold values.
    Rincdec : float, default=0.005
        Increase/Decrease step for threshold values.
    noiseRemovalThresholdFacBG : float, default=0.0004
        Strength of the noise removal for background points.
    noiseRemovalThresholdFacFG : float, default=0.0008
        Strength of the noise removal for foreground points.
    LSBPthreshold : int, default=8
        Threshold for LSBP binary string.
    minCount : int, default=2
        Minimal number of matches for sample to be considered as foreground.

    """

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