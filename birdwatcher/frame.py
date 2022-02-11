import numpy as np

#TODO decide if gray frame has a 3rd dimension of size 1, or not
# opencv has 2D for gray, and 3D for color
class Frame:
    """A video frame

    The underlying image data is stored in a 2D (height, width) or 3D (height,
    width, channel) numpy array.

    """
    def __init__(self, array, dtype=None):

        self._array = np.asarray(array, dtype)
        self._dtype = str(self._array.dtype)
        self._frameheight = self._array.shape[0]
        self._framewidth = self._array.shape[1]
        if len(self._array.shape) == 2:
            self._isgray = True
            self._nchannels = 0
        else:
            self._isgray = False
            self._nchannels = self._framewidth = self._array.shape[2]

    @property
    def array(self):
        return self._array

    @property
    def frameheight(self):
        return self._frameheight

    @property
    def framewidth(self):
        return self._framewidth

    @property
    def nchannels(self):
        return self._nchannels

    @property
    def dtype(self):
        return self._dtype

