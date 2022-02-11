"""This module contains classes for generating image frames and general
processing functionality such as measurement, drawing of labels/text and
saving as video files.

Many methods mirror functions from OpenCV. Doc strings are provided but
it is a good idea to look at OpenCV's documentation and examples if you want
to understand the parameters in more depth.

"""

import numpy as np
import cv2 as cv
from functools import wraps

from .utils import peek_iterable

__all__ = ['Frames', 'FramesColor', 'FramesGray', 'FramesNoise', 'framecolor',
           'framegray', 'framenoise']

def frameiterator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        processingdata = {}
        self = args[0]
        if hasattr(self, 'get_info'):
            processingdata['processingdata'] = self.get_info()
            processingdata['methodname'] = func.__name__
            processingdata['methodargs'] = [str(arg) for arg in args]
            processingdata['methodkwargs'] = dict((str(key),str(item))
                                                  for (key, item)
                                                  in kwargs.items())
            processingdata['classname'] = self.__class__.__name__

        return Frames(func(*args, **kwargs), processingdata=processingdata)
    return wrapper

#TODO add some way of easily starting and stopping at artbitrary frame numbers
class Frames:
    """An iterator of video frames with useful methods.

    This is a main base class in Birdwatcher, as many functions and
    methods return this type and take it as input. Many methods of `Frames`
    objects return new `Frames` objects, but some of them generate final
    output, such as a video file or a measurement.

    Parameters
    ----------
    frames: iterable
        This can be anything that is iterable and that produces image frames. A
        numpy array, a VideoFileStream or another Frames object.

    Examples
    --------
    >>> import birdwatcher as bw
    >>> frames = bw.FramesNoise(250, height=720, width=1280)
    >>> frames = frames.draw_framenumbers()
    >>> frames.tovideo('noisewithframenumbers.mp4', framerate=25)
    >>> # next example based on input from video file
    >>> vf = bw.VideoFileStream('zebrafinchrecording.mp4')
    >>> frames = vf.iter_frames() # create Frames object
    >>> # more concise expression
    >>> frames.blur(ksize=(3,3)).togray().tovideo('zf_blurgray.mp4')

    """

    def __init__(self, frames, processingdata=None):

        first, frames = peek_iterable(frames)

        framewidth, frameheight, *nchannels = first.shape
        if nchannels == []:
            nchannels = 1
        self._frames = frames
        self._frameheight = frameheight
        self._framewidth = framewidth
        self._nchannels = nchannels
        self._index = 0
        self._dtype = first.dtype.name
        self.processingdata = processingdata

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._frames)

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

    def get_info(self):
        return {'classname': self.__class__.__name__,
                'framewidth': self.framewidth,
                'frameheight': self.frameheight,
                'processingdata': self.processingdata}

    def tovideo(self, filename, framerate, crf=23, scale=None, format='mp4',
                codec='libx264', pixfmt='yuv420p', ffmpegpath='ffmpeg'):
        """Writes frames to video file.

        Parameters
        ----------
        filename: str
            Name of the videofile that should be written to
        framerate: int
            framerate of video in frames per second
        crf: int
            Value determines quality of video. Default: 23, which is good
            quality. See ffmpeg documentation. Use 17 for high quality.
        scale: tuple or None
            (width, height). If None, do not change width and height.
            Default: None.
        format: str
            ffmpeg video format. Default is 'mp4'. See ffmpeg documentation.
        codec: str
            ffmpeg video codec. Default is 'libx264'. See ffmpeg documentation.
        pixfmt: str
            ffmpeg pixel format. Default is 'yuv420p'. See ffmpeg documentation.
        ffmpegpath: str or pathlib.Path
            Path to ffmpeg executable. Default is `ffmpeg`, which means it
            should be in the system path.

        """
        from .ffmpeg import arraytovideo
        from .video import VideoFileStream
        filepath = arraytovideo(frames=self, filename=filename,
                                framerate=framerate, crf=crf, scale=scale,
                                format=format, codec=codec, pixfmt=pixfmt,
                                ffmpegpath=ffmpegpath)
        return VideoFileStream(filepath)


    @frameiterator
    def blur(self, ksize, anchor=(-1,-1), borderType=cv.BORDER_DEFAULT):
        """Blurs frames using the normalized box filter.

        Parameters
        ----------
        ksize: (int, int)
            Kernel size. Tuple of ints.
        anchor: (int, int)
            Anchor point. Default value (-1,-1) means that the anchor is at
            the kernel center.
        borderType: int
            Border mode used to extrapolate pixels outside of the image.
            Default: 4.

        Returns
        -------
        Frames
            Iterator that generates blurred frames.

        Examples
        --------
        >>> import birdwatcher as bw
        >>> frames = bw.FramesNoise(250, height=720, width=1280)
        >>> frames = frames.blur(ksize=(10,10))
        >>> frames.tovideo('noiseblurred.mp4', framerate=25)

        """
        for frame in self._frames:
            yield cv.blur(frame, ksize=ksize, anchor=anchor,
                          borderType=borderType)

    # FIXME multiple circles per frame?
    @frameiterator
    def draw_circles(self, centers, radius=6, color=(255, 100, 0),
                     thickness=2, linetype=cv.LINE_AA, shift=0):
        """Draws circles on frames.

        Centers should be an iterable that has a length that corresponds to
        the number of frames.

        Parameters
        ----------
        centers: iterable
            Iterable that generate center coordinates (x, y) of the circles
        radius: int
            Radius of circle. Default 4.
        color: tuple of ints
            Color of circle (r, g, b). Default (255, 100, 0)
        thickness: int
            Line thickness. Default 2.
        linetype: int
            OpenCV line type of circle boundary. Default cv2.LINE_AA
        shift: int
            Number of fractional bits in the coordinates of the center and in
            the radius value. Default 0.

        Returns
        -------
        Frames
            Iterator that generates frames with circles

        """

        for frame, center in zip(self._frames, centers):
            center = np.asanyarray(center)
            if not np.isnan(center).any():
                (x, y) = center.astype('int16')
                yield cv.circle(frame, center=(x, y), radius=radius,
                                color=color,
                                thickness=thickness, lineType=linetype,
                                shift=shift)
            else:
                yield frame

    @frameiterator
    def draw_rectangles(self, points, color=(255, 100, 0),
                     thickness=2, linetype=cv.LINE_AA, shift=0):
        """Draws circles on frames.

        Centers should be an iterable that has a length that corresponds to
        the number of frames.

        Parameters
        ----------
        points: iterable
            Iterable that generates sequences of rectangle corners ((x1, y1),
            (x2, y2)) per frame.
        color: tuple of ints
            Color of rectangle (r, g, b). Default (255, 100, 0)
        thickness: int
            Line thickness. Default 2.
        linetype: int
            OpenCV line type of rectangle boundary. Default cv2.LINE_AA
        shift: int
            Number of fractional bits in the  point coordinates. Default 0.

        Returns
        -------
        Frames
            Iterator that generates frames with rectangles

        """

        for frame, framepoints in zip(self._frames, points):
            for (pt1, pt2) in framepoints:
                yield cv.rectangle(frame, pt1=pt1, pt2=pt2, color=color,
                                thickness=thickness, lineType=linetype,
                                shift=shift)
            else:
                yield frame

    #FIXME check if input is color
    @frameiterator
    def togray(self):
        """Converts color frames to gray frames

        Returns
        -------
        Frames

        """
        for frame in self._frames:
            yield cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # FIXME check if input is gray
    @frameiterator
    def tocolor(self):
        """Converts gray frames to color frames

        Returns
        -------
        Frames

        """
        for frame in self._frames:
            yield cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    # FIXME use draw_text
    @frameiterator
    def draw_framenumbers(self, startat=0, org=(2, 25),
                          fontface=cv.FONT_HERSHEY_SIMPLEX,
                          fontscale=1, color=(200, 200, 200), thickness=2,
                          linetype=cv.LINE_AA):
        """Draws the frame number on frames.

        Parameters
        ----------
        startat: int
            The number to start counting at. Default 0
        org: 2-tuple of ints
            Bottom-left corner of the text string in the image.
        fontface: OpenCV font type
            Default cv.FONT_HERSHEY_SIMPLEX
        fontscale: float
            Font scale factor that is multiplied by the font-specific base
            size.
        color: tuple of ints
            Color of circle (r, g, b). Default (255, 100, 0)
        thickness: int
            Line thickness. Default 2.
        linetype: int
            OpenCV line type of circle boundary. Default cv2.LINE_AA

        Returns
        -------
        iterator
            Iterator that generates frames with frame numbers

        """
        for frameno, frame in enumerate(self._frames):
            yield cv.putText(frame, str(frameno + startat), org=org,
                             fontFace=fontface, fontScale=fontscale,
                             color=color, thickness=thickness,
                             lineType=linetype)

    @frameiterator
    def draw_text(self, textiterator, org=(2, 25),
                  fontface=cv.FONT_HERSHEY_SIMPLEX, fontscale=1,
                  color=(200, 200, 200), thickness=2, linetype=cv.LINE_AA):
        """Draws the frame number on frames.

            Parameters
            ----------
            textiterator: iterable
                Someting that you can iterate over and that produces text
                for each frame
            fontface: OpenCV font type
                Default cv.FONT_HERSHEY_SIMPLEX
            fontscale: float
                Font scale factor that is multiplied by the font-specific base
                size.
            color: tuple of ints
                Color of circle (r, g, b). Default (200, 200, 200)
            thickness: int
                Line thickness. Default 2.
            linetype: int
                OpenCV line type of circle boundary. Default cv2.LINE_AA

            Returns
            -------
            iterator
                Iterator that generates frames with frame numbers

        """

        for frame, text in zip(self._frames, textiterator):
            yield cv.putText(frame, str(text), org=org,
                             fontFace=fontface, fontScale=fontscale,
                             color=color, thickness=thickness,
                             lineType=linetype)

    # FIXME should this be a coordinate iterator?
    @frameiterator
    def find_nonzero(self):
        """Yields the locations of non-zero pixels.

        Returns
        -------
        Iterator
            Iterates over a sequence of shape (N, 2) arrays, where N is the
            number of frames.

        """
        for frame in self._frames:
            idx = cv.findNonZero(frame)
            if idx is None:
                idx = np.zeros((0,2), dtype=np.uint16)
            else:
                idx = idx[:, 0, :]
            yield idx

    @frameiterator
    def morphologyex(self, morphtype='open', kernelsize=2, iterations=1):
        """Performs advanced morphological transformations on frames.

        Can perform advanced morphological transformations using an erosion
        and dilation as basic operations.

        In case of multi-channel images, each channel is processed
        independently.

        Parameters
        ----------
        morphtype: str
            Type of transformation. Choose from 'erode', 'dilate', 'open',
            'close', 'gradient', 'tophat', 'blackhat'. Default: 'open'.
        kernelsize: int
            Size of kernel in 1 dimension. Default 2.
        iterations: int
            Number of times erosion and dilation are applied.

        Returns
        -------
        Frames
            Iterates over sequence of transformed image frames.

        """
        morphtypes={'erode': cv.MORPH_ERODE,
                    'dilate': cv.MORPH_DILATE,
                    'open': cv.MORPH_OPEN,
                    'close': cv.MORPH_CLOSE,
                    'gradient': cv.MORPH_GRADIENT,
                    'tophat': cv.MORPH_TOPHAT,
                    'blackhat': cv.MORPH_BLACKHAT}
        morphnum = morphtypes.get(morphtype, None)
        if morphnum is None:
            raise ValueError(f'`{morphtype}` is not a valid morphtype')
        kernel = np.ones((kernelsize, kernelsize), np.uint8)
        for frame in self._frames:
            yield cv.morphologyEx(frame, morphnum, kernel,
                                  iterations=iterations)

    @frameiterator
    def add_weighted(self, alpha, frames, beta, gamma=0):
        """Calculates the weighted sum of frames from self and the frames
        of the object specified by the `frames` parameter.

        Parameters
        ----------
        alpha: float
            Weight of the frames of self.
        frames: frame iterator
            The other source of input frames.
        beta: float
            Weight of the frames of the other frame iterator, specified by
            the `frames` parameter.
        gamma: float
            Scalar added to each sum.

        Returns
        -------
        Frames
            Iterates over sequence of summed image frames.

        """
        for frame1, frame2 in zip(self._frames, frames):
            yield cv.addWeighted(src1=frame1, alpha=alpha, src2=frame2,
                                 beta=beta, gamma=gamma)

    @frameiterator
    def apply_backgroundsegmenter(self, bgs, fgmask=None, learningRate=-1.0,
                                  roi=None, nroi=None):
        """Compute foreground masks based on input sequence of frames.

        Parameters
        ----------
        bgs: Subclass of BaseBackgroundSubtractor
            Instance of one of Birdwatcher's BackgroundSubtractor classes,
            such as BackgroundSubtractorMOG2.
        fgmask: numpy array image
            The output foreground mask as an 8-bit binary image.
        learningRate: float
            The value between 0 and 1 that indicates how fast the background
            model is learnt. Negative parameter value makes the algorithm to
            use some automatically chosen learning rate. 0 means that the
            background model is not updated at all, 1 means that the background
            model is completely reinitialized from the last frame.
        roi: (int, int, int, int) or None
            Region of interest. Only look at this rectangular region. h1,
            h2, w1, w2. Default None.

        Returns
        -------
        Frames
            Iterates over sequence of foreground masks.

        """
        return bgs.iter_apply(self._frames, fgmask=fgmask,
                              learningRate=learningRate, roi=roi,
                              nroi=nroi)

    @frameiterator
    def crop(self, h1, h2, w1, w2):
        """Crops frames to a smaller size.

        Parameters
        ----------
        h1: int
            Top pixel rows
        h2: int
            Bottom pixel row
        w1: int
            Left pixel column
        w2: int
            Right pixel colum

        Returns
        -------
        Frames
            Iterates over sequence of cropped frames.

        """
        for frame in self._frames:
            yield frame[h1:h2, w1:w2]

    @frameiterator
    def absdiff_frame(self, frame):
        """Subtract static image frame from frame iterator.

        Parameters
        ----------
        frame: ndarray frame
            Fixed image frame that will be subtracted from each frame of the
            frame iterator.

        Returns
        -------
        Frames
            Iterates over sequence of absolute difference frames.

        """
        for frame_self in self._frames:
            yield cv.absdiff(frame_self, frame)

    @frameiterator
    def threshold(self, thresh,	maxval=255,	threshtype='tozero'):
        """Thresholds frames at value `thresh`.

        Parameters
        ----------
        thresh: int
        	threshold value.
        maxval: int
        	Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV
        	thresholding types. Default 255
        threshtype: int
        	Thresholding type. Choose from 'binary', 'binary_inv', 'trunc',
            'tozero', 'tozero_inv', 'mask', 'otsu', 'triangle'. Default:
            'tozero', which means that everything below `thresh` will be set
            to zero. See doc OpenCV.

        Returns
        -------
        Frames
            Iterates over sequence of thresholded frames.

        """
        threshtypes = {
        'binary': cv.THRESH_BINARY,
        'binary_inv': cv.THRESH_BINARY_INV,
        'trunc': cv.THRESH_TRUNC,
        'tozero': cv.THRESH_TOZERO,
        'tozero_inv': cv.THRESH_TOZERO_INV,
        'mask': cv.THRESH_MASK,
        'otsu': cv.THRESH_OTSU,
        'triangle': cv.THRESH_TRIANGLE
        }
        threshtype = threshtypes[threshtype]
        for frame in self._frames:
            yield cv.threshold(src=frame, thresh=thresh,
                               maxval=maxval, type=threshtype)[1]

    @frameiterator
    def resize(self, dsize, interpolation='linear'):
        """Resizes frames.

        Parameters
        ----------
        dsize: tuple
            Destination size (width, height).
        interpolation: str
            interpolation method:
            nearest - a nearest-neighbor interpolation
            linear - a bilinear interpolation (used by default)
            area - resampling using pixel area relation. It may be a preferred
                method for image decimation, as it gives moire-free
                results. But when the image is zoomed, it is similar to
                the nearest method.
            cubic - a bicubic interpolation over 4x4 pixel neighborhood
            lanczos4 - a Lanczos interpolation over 8x8 pixel neighborhood

        Returns
        -------
        Frames
            Iterates over sequence of thresholded frames.

        """
        interptypes = {
            'nearest': cv.INTER_NEAREST,
            'linear': cv.INTER_LINEAR,
            'area': cv.INTER_AREA,
            'cubic': cv.INTER_CUBIC,
            'lanczos4': cv.INTER_LANCZOS4
        }
        interpolation = interptypes[interpolation]
        for frame in self._frames:
            yield cv.resize(src=frame, dsize=dsize, fx=0, fy=0,
                            interpolation=interpolation)

    @frameiterator
    def resizebyfactor(self, fx, fy, interpolation='linear'):
        """Resizes frames by a specified factor.

        Parameters
        ----------
        fx: float
            Scale factor along the horizontal axis.
        fy: float
            Scale factor along the vertical axis.
        interpolation: str
            interpolation method:
            nearest - a nearest-neighbor interpolation
            linear - a bilinear interpolation (used by default)
            area - resampling using pixel area relation. It may be a preferred
                method for image decimation, as it gives moire’-free
                results. But when the image is zoomed, it is similar to
                the nearest method.
            cubic - a bicubic interpolation over 4x4 pixel neighborhood
            lanczos4 - a Lanczos interpolation over 8x8 pixel neighborhood

        Returns
        -------
        Frames
            Iterates over sequence of thresholded frames.

        """
        interptypes = {
            'nearest': cv.INTER_NEAREST,
            'linear': cv.INTER_LINEAR,
            'area': cv.INTER_AREA,
            'cubic': cv.INTER_CUBIC,
            'lanczos4': cv.INTER_LANCZOS4
        }
        interpolation = interptypes[interpolation]
        for frame in self._frames:
            yield cv.resize(src=frame, dsize=(0,0), fx=fx, fy=fy,
                            interpolation=interpolation)

    # FIXME should this be a coordinate iterator?
    def find_contours(self, retrmode='tree', apprmethod='simple',
                      offset=(0, 0)):
        """Finds contrours in frames.

        Parameters
        ----------
        retrmode:  str

        apprmethod: str

        offset: (int, int)

        Returns
        -------
        Iterator
            Iterates over contours.

        """
        retrmode = {
            'tree': cv.RETR_TREE,
            'external': cv.RETR_EXTERNAL,
            'list': cv.RETR_LIST,
            'ccomp': cv.RETR_CCOMP,
            'floodfill': cv.RETR_FLOODFILL
        }[retrmode]
        apprmethod ={
            'none': cv.CHAIN_APPROX_NONE,
            'simple': cv.CHAIN_APPROX_SIMPLE,
            'tc89_l1': cv.CHAIN_APPROX_TC89_L1,
            'tc89_kcos': cv.CHAIN_APPROX_TC89_KCOS
        }[apprmethod]
        for frame in self._frames:
            yield cv.findContours(frame, mode=retrmode,
                                  method=apprmethod, offset=offset)

    def calc_meanframe(self, dtype=None):
        if self.nchannels == 1:
            meanframe = framegray(self.frameheight, self.framewidth, value=0, dtype='float64')
        else:
            meanframe = framecolor(self.frameheight, self.framewidth, color=(0, 0, 0), dtype='float64')
        for i, frame in enumerate(self._frames):
            meanframe += frame
        meanframe /= i
        if dtype is None:
            dtype = self._dtype
        return meanframe.astype(dtype)

    # TODO Frames should have a frame rate
    # TODO Exception handling
    def show(self, framerate=25):
        """Shows frames in a video window.

        Parameters
        ----------
        framerate

        Returns
        -------

        """

        waitkey = int(round(1000 / framerate))
        for frame in self._frames:
            cv.imshow('frame', frame)
            if cv.waitKey(waitkey) & 0xFF == ord('q'):
                break
        cv.destroyAllWindows()


class FramesColor(Frames):
    """An iterator that yields color frames.

    This class inherits from Frames, and hence has all its methods.

    """

    def __init__(self, nframes, width, height, color=(0, 0, 0), dtype='uint8'):
        """Creates an iterator that yields color frames.

        Parameters
        ----------
        nframes: int
            Number of frames to be produced.
        width: int
            Width of frame.
        height: int
            Height of frame.
        color:
            Fill value of frame. Default (0, 0, 0) (black).
        dtype: numpy dtype
            Dtype of frame. Default `uint8'

        Returns
        -------
        Iterator of numpy ndarrays

        """
        frame = framecolor(width=width, height=height, color=color,
                           dtype=dtype)
        frames = (frame.copy() for _ in range(nframes))
        super().__init__(frames=frames)


class FramesGray(Frames):
    """An iterator that yields gray frames.

    This class inherits from Frames, and hence has all its methods.

    """

    def __init__(self, nframes, width, height, value=0, dtype='uint8'):
        """Creates an iterator that yields gray frames.

        Parameters
        ----------
        nframes: int
            Number of frames to be produced.
        width: int
            Width of frame.
        height: int
            Height of frame.
        value:
            Fill value of frame. Default 0 (black).
        dtype: numpy dtype
            Dtype of frame. Default `uint8'

        Returns
        -------
        Iterator of numpy ndarrays

        """

        frame = framegray(width=width, height=height, value=value,
                          dtype=dtype)

        frames = (frame.copy() for _ in range(nframes))
        super().__init__(frames=frames)


class FramesNoise(Frames):
    """An iterator that yields noise frames.

    This class inherits from Frames, and hence has all its methods.

    """

    def __init__(self, nframes,width,  height, dtype='uint8'):
        """Creates an iterator that yields gray frames.

        Parameters
        ----------
        nframes: int
            Number of frames to be produced.
        width: int
            Width of frame.
        height: int
            Height of frame.
        dtype: numpy dtype
            Dtype of frame. Default `uint8'

        Returns
        -------
        Iterator of numpy ndarrays

        """

        frames = (framenoise(height=height, width=width, dtype=dtype)
                  for _ in range(nframes))
        super().__init__(frames=frames)


def framegray(width, height, value=0, dtype='uint8'):
    """Creates a gray frame.

    Parameters
    ----------
    width: int
        Width of frame.
    height: int
        Height of frame.
    value:
        Fill value of frame. Default 0 (black).
    dtype: numpy dtype
        Dtype of frame. Default `uint8'

    Returns
    -------
    numpy ndarray

    """
    return np.ones((height, width), dtype=dtype) * value


def framecolor(width, height, color=(0, 0, 0), dtype='uint8'):
    """Creates a color frame.

    Parameters
    ----------
    width: int
        Width of frame.
    height: int
        Height of frame.
    color:
        Fill value of frame. Default (0, 0, 0) (black).
    dtype: numpy dtype
        Dtype of frame. Default `uint8'

    Returns
    -------
    numpy ndarray

    """
    return np.ones((height, width, 3), dtype=dtype) * np.asanyarray(color,
                                                                    dtype=dtype)


def framenoise(width, height, dtype='uint8'):
    """Creates a noise frame.

    Parameters
    ----------
    width: int
        Width of frame.
    height: int
        Height of frame.
    dtype: numpy dtype
        Dtype of frame. Default `uint8'

    Returns
    -------
    numpy ndarray

    """

    return np.random.randint(0, 255, (height, width, 3), dtype=dtype)


def create_frameswithmovingcircle(nframes, width, height, framecolor=(0, 0, 0),
                                  circlecolor=(255, 100, 0), radius=6,
                                  thickness=2, linetype=8, dtype='uint8'):
    frames = FramesColor(nframes=nframes,  width=width, height=height,
                         color=framecolor, dtype=dtype)
    centers = zip(np.linspace(0, width, nframes),
                  np.linspace(0, height, nframes))
    return frames.draw_circles(centers, color=circlecolor, radius=radius,
                               thickness=thickness, linetype=linetype)
