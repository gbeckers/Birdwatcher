"""This module contains classes for generating image frames and general
processing functionality such as measurement, drawing of labels/text and
saving as video files.

Many methods mirror functions from OpenCV. Docstrings are provided but
it is a good idea to look at OpenCV's documentation and examples if you want
to understand the parameters in more depth.

"""

from functools import wraps
from pathlib import Path

import numpy as np
import cv2 as cv

from .utils import peek_iterable


__all__ = ['Frames', 'FramesColor', 'FramesGray', 'FramesNoise', 'framecolor',
           'framegray', 'framenoise']


def _check_writable(frame):
    if not frame.flags.writeable:
        return frame.copy()
    else:
        return frame


def frameiterator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        processingdata = []
        self = args[0]
        if hasattr(self, 'get_info'):
            processingdata = self.get_info().get('processingdata') or []
        processingdata.append({'methodname': func.__name__,
                               'methodargs': [str(arg) for arg in args],
                               'methodkwargs': dict((str(key),str(item))
                                                    for (key, item)
                                                    in kwargs.items())})
        return Frames(func(*args, **kwargs), processingdata=processingdata)
    return wrapper

class Frames:
    """An iterator of video frames with useful methods.

    This is a main base class in Birdwatcher, as many functions and
    methods return this type and take it as input. Many methods of `Frames`
    objects return new `Frames` objects, but some of them generate final
    output, such as a video file or a measurement.

    Parameters
    ----------
    frames : iterable
        This can be anything that is iterable and that produces image frames.
        A numpy array, a VideoFileStream or another Frames object.

    Examples
    --------
    >>> import birdwatcher as bw
    >>> frames = bw.FramesNoise(250, height=720, width=1280)
    >>> frames = frames.draw_framenumbers()
    >>> frames.tovideo('noisewithframenumbers.mp4', framerate=25)
    >>> # next example based on input from video file
    >>> vfs = bw.VideoFileStream('zebrafinchrecording.mp4')
    >>> frames = vfs.iter_frames() # create Frames object
    >>> # more concise expression
    >>> frames.blur(ksize=(3,3)).togray().tovideo('zf_blurgray.mp4')

    """

    def __init__(self, frames, processingdata=None):

        first, frames = peek_iterable(frames)

        frameheight, framewidth, *nchannels = first.shape
        if nchannels == []:
            nchannels = [1]
        self._frames = frames
        self._frameheight = frameheight
        self._framewidth = framewidth
        self._nchannels = nchannels[0]
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
    
    def peek_frame(self):
        """Returns first frame without removing it.
        
        Returns
        -------
        numpy ndarray
            The first frame.
        """
        firstframe, self._frames = peek_iterable(self._frames)
        return firstframe
        
    def tovideo(self, filepath, framerate, crf=23, scale=None, format='mp4',
                codec='libx264', pixfmt='yuv420p', ffmpegpath='ffmpeg'):
        """Writes frames to video file.

        Parameters
        ----------
        filepath : str
            Name of the videofilepath that should be written to.
        framerate : int
            framerate of video in frames per second.
        crf : int, default=23
            Value determines quality of video. The default 23 is good quality.
            Use 17 for high quality.
        scale : tuple, optional
            (width, height). The default (None) does not change width and    
            height.
        format : str, default='mp4'
            ffmpeg video format.
        codec : str, default='libx264'
            ffmpeg video codec.
        pixfmt : str, default='yuv420p'
            ffmpeg pixel format.
        ffmpegpath : str or pathlib.Path, optional
            Path to ffmpeg executable. Default is `ffmpeg`, which means it
            should be in the system path.

        Notes
        -----
        See ffmpeg documentation for more information.

        """
        from .ffmpeg import arraytovideo
        from .video import VideoFileStream
        filepath = arraytovideo(frames=self, filepath=filepath,
                                framerate=framerate, crf=crf, scale=scale,
                                format=format, codec=codec, pixfmt=pixfmt,
                                ffmpegpath=ffmpegpath)
        return VideoFileStream(filepath)

    @frameiterator
    def blur(self, ksize, anchor=(-1,-1), borderType=cv.BORDER_DEFAULT):
        """Blurs frames using the normalized box filter.

        Parameters
        ----------
        ksize : (int, int)
            Kernel size. Tuple of integers.
        anchor : (int, int), optional
            Anchor point. Default value (-1,-1) means that the anchor is at
            the kernel center.
        borderType : int, default=cv.BORDER_DEFAULT
            Border mode used to extrapolate pixels outside of the image.

        Yields
        ------
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

    @frameiterator
    def edge_detection(self, minval=80, maxval=150):
        """Finds edges (boundaries) in frames.

        Only works on gray frames! Blur frames before applying edge detection
        for optimal results. Edges are defined by sudden changes in pixel
        intensity.

        Parameters
        ----------
        minval : str, optional
            Lower threshold for finding smaller edges.
        maxval : str, optional
            Higher threshold to determine segments of strong edges.

        Yields
        ------
        Frames
            Iterator that generates frames with edges.

        """
        for frame in self._frames:
            yield cv.Canny(frame, minval, maxval)

    # FIXME multiple circles per frame?
    @frameiterator
    def draw_circles(self, centers, radius=6, color=(255, 100, 0),
                     thickness=2, linetype=cv.LINE_AA, shift=0):
        """Draws circles on frames.

        Centers should be an iterable that has a length that corresponds to
        the number of frames.

        Parameters
        ----------
        centers : iterable
            Iterable that generate center coordinates (x, y) of the circles
        radius : int, default=6
            Radius of circle.
        color : tuple of ints, optional
            Color of circle (BGR). The default (255, 100, 0) color is blue.
        thickness : int, default=2
            Line thickness.
        linetype : int, default=cv2.LINE_AA
            OpenCV line type of circle boundary.
        shift : int, default=0
            Number of fractional bits in the coordinates of the center and in
            the radius value.

        Yields
        ------
        Frames
            Iterator that generates frames with circles.

        """
        for frame, center in zip(self._frames, centers):
            frame = _check_writable(frame)
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
    def draw_rectangles(self, points, color=(255, 100, 0), thickness=2):
        """Draws rectangles on frames.

        Points should be an iterable that has a length that corresponds to
        the number of frames.

        Parameters
        ----------
        points : iterable
            Iterable that generates sequences of rectangle corners ((x1, y1),
            (x2, y2)) per frame, where the coordinates specify opposite
            corners (e.g. top-left and bottom-right).
        color : tuple of ints, optional
            Color of rectangle (BGR). The default (255, 100, 0) color is blue.
        thickness : int, default=2
            Line thickness.

        Yields
        ------
        Frames
            Iterator that generates frames with rectangles.

        """
        for frame, framepoints in zip(self._frames, points):
            frame = _check_writable(frame)
            framepoints = np.asanyarray(framepoints)
            if not np.isnan(framepoints).any():
                (pt1, pt2) = framepoints.astype('int16')
                yield cv.rectangle(frame, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
            else:
                yield frame

    @frameiterator
    def togray(self):
        """Converts color frames to gray frames using OpenCV.

        Yields
        ------
        Frames
            Iterator that generates gray frames.

        """
        for frame in self._frames:
            if frame.ndim == 3:
                yield cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            else:
                yield frame

    @frameiterator
    def tocolor(self):
        """Converts gray frames to color frames using OpenCV.

        Yields
        ------
        Frames
            Iterator that generates color frames.

        """
        for frame in self._frames:
            if frame.ndim == 2:
                yield cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            else:
                yield frame

    @frameiterator
    def draw_framenumbers(self, startat=0, org=(2, 25),
                          fontface=cv.FONT_HERSHEY_SIMPLEX,
                          fontscale=1, color=(200, 200, 200), thickness=2,
                          linetype=cv.LINE_AA):
        """Draws the frame number on frames.

        Parameters
        ----------
        startat : int, optional
            The number to start counting at.
        org : (int, int), optional
            A tuple of ints (horizontal coordinate value, vertical coordinate
            value) indicates where to draw the framenumbers. The default (2,
            25) draws numbers in the top left corner of the image.
        fontface : OpenCV font type, default=cv.FONT_HERSHEY_SIMPLEX
        fontscale : float, optional
            Font scale factor that is multiplied by the font-specific base
            size.
        color : (int, int, int), optional
            Font color (BGR). The default (200, 200, 200) color is gray.
        thickness : int, default=2
            Line thickness.
        linetype : int, default=cv2.LINE_AA
            OpenCV line type.

        Yields
        ------
        Frames
            Iterator that generates frames with frame numbers.

        """
        for frameno, frame in enumerate(self._frames):
            frame = _check_writable(frame)
            yield cv.putText(frame, str(frameno + startat), org=org,
                             fontFace=fontface, fontScale=fontscale,
                             color=color, thickness=thickness,
                             lineType=linetype)

    @frameiterator
    def draw_text(self, textiterator, org=(2, 25),
                  fontface=cv.FONT_HERSHEY_SIMPLEX, fontscale=1,
                  color=(200, 200, 200), thickness=2, linetype=cv.LINE_AA):
        """Draws text on frames.

            Parameters
            ----------
            textiterator : iterable
                Something that you can iterate over and that produces text
                for each frame.
            org : (int, int), optional
                A tuple of ints (horizontal coordinate value, vertical
                coordinate value) indicates where to draw the text. The
                default (2, 25) draws text in the top left corner of the
                image.
            fontface : OpenCV font type, default=cv.FONT_HERSHEY_SIMPLEX
            fontscale : float, optional
                Font scale factor that is multiplied by the font-specific base
                size.
            color : (int, int, int), optional
                Font color (BGR). The default color (200, 200, 200) is gray.
            thickness : int, default=2
                Line thickness.
            linetype : int, default=cv2.LINE_AA
                OpenCV line type.

            Yields
            ------
            Frames
                Iterator that generates frames with text.

        """
        for frame, text in zip(self._frames, textiterator):
            frame = _check_writable(frame)
            yield cv.putText(frame, str(text), org=org,
                             fontFace=fontface, fontScale=fontscale,
                             color=color, thickness=thickness,
                             lineType=linetype)
        
    def save_nonzero(self, filepath, metadata, ignore_firstnframes=10, 
                     overwrite=True):
        """Save nonzero pixel coordinates (i.e. foreground) as Coordinate 
        Arrays object.

        Parameters
        ----------
        filepath : str
            Name of the filepath that should be written to.
        metadata : dict, optional
        ignore_firstnframes : int, default=10
            Do not provide coordinates for the first n frames. These often 
            have a lot of false positives.
        overwrite : bool, default=True
             Overwrite existing CoordinateArrays or not.
        
        Returns
        -------
        CoordinateArrays  
        
        """
        from .coordinatearrays import create_coordarray
        
        if Path(filepath).suffix != '.darr':
            filepath = filepath + '.darr'
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        coordsarray = create_coordarray(filepath,
                                        framewidth=self._framewidth,
                                        frameheight=self._frameheight,
                                        metadata=metadata,
                                        overwrite=overwrite)
        
        empty = np.zeros((0,2), dtype=np.uint16)
        coords = (c if i >= ignore_firstnframes else empty for i,c in
                  enumerate(self.find_nonzero()))
        coordsarray.iterappend(coords)
        
        return coordsarray

    def find_nonzero(self):
        """Yields the locations of non-zero pixels.

        If the frame is a color frame, non-zero means that a pixel
        does not have the value (0,0,0).

        Yields
        ------
        Iterator that generates shape (N, 2) arrays, where N is the number
            of non-zero pixels.

        """
        for frame in self._frames:
            if frame.ndim == 3:
                frame = (frame!=0).sum(axis=2, dtype=frame.dtype)
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
        morphtype : {'open', 'erode', 'dilate, 'close', 'gradient', 'tophat',
        'blackhat'}
            Type of transformation. Default is 'open', which is an erosion
            followed by a dilation.
        kernelsize : int, default=2
            Size of kernel in 1 dimension.
        iterations : int, default=1
            Number of times erosion and dilation are applied.

        Yields
        ------
        Frames
            Iterator that generates transformed image frames.

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
        alpha : float
            Weight of the frames of self.
        frames : frame iterator
            The other source of input frames.
        beta : float
            Weight of the frames of the other frame iterator, specified by
            the `frames` parameter.
        gamma : float, optional
            Scalar added to each sum.

        Yields
        ------
        Frames
            Iterator that generates summed image frames.

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
        bgs : Subclass of BaseBackgroundSubtractor
            Instance of one of Birdwatcher's BackgroundSubtractor classes,
            such as BackgroundSubtractorMOG2.
        fgmask : numpy array image, optional
            The output foreground mask as an 8-bit binary image.
        learningRate : float, optional
            The value between 0 and 1 that indicates how fast the background
            model is learnt. The default negative parameter value (-1.0) makes
            the algorithm to use some automatically chosen learning rate. 0
            means that the background model is not updated at all, 1 means
            that the background model is completely reinitialized from the
            last frame.
        roi : (int, int, int, int), optional
            Region of interest. Only look at this rectangular region. h1,
            h2, w1, w2.
        nroi : (int, int, int, int), optional
            Not region of interest. Exclude this rectangular region. h1,
            h2, w1, w2.

        Yields
        ------
        Frames
            Iterator that generates foreground masks.

        """
        if roi is not None:
            firstframe = self.peek_frame()
            completeframe = np.zeros((firstframe.shape[0],
                                      firstframe.shape[1]), dtype=np.uint8)
        for frame in self._frames:
            if roi is not None:
                h1,h2,w1,w2 = roi
                frame = frame[h1:h2, w1:w2]
            mask = bgs.apply(frame=frame, fgmask=fgmask,
                             learningRate=learningRate)
            if roi is not None:
                completeframe[h1:h2, w1:w2] = mask
                mask = completeframe
            if nroi is not None:
                h1,h2,w1,w2 = nroi
                mask[h1:h2, w1:w2] = 0
            yield mask

    @frameiterator
    def crop(self, h1, h2, w1, w2):
        """Crops frames to a smaller size.

        Parameters
        ----------
        h1 : int
            Top pixel rows.
        h2 : int
            Bottom pixel row.
        w1 : int
            Left pixel column.
        w2 : int
            Right pixel colum.

        Yields
        ------
        Frames
            Iterator that generates cropped frames.

        """
        for frame in self._frames:
            yield frame[h1:h2, w1:w2]

    @frameiterator
    def absdiff_frame(self, frame):
        """Subtract static image frame from frame iterator.

        Parameters
        ----------
        frame : ndarray frame
            Fixed image frame that will be subtracted from each frame of the
            frame iterator.

        Yields
        ------
        Frames
            Iterator that generates absolute difference frames.

        """
        for frame_self in self._frames:
            yield cv.absdiff(frame_self, frame)

    @frameiterator
    def threshold(self, thresh,	maxval=255,	threshtype='tozero'):
        """Thresholds frames at value `thresh`.

        Parameters
        ----------
        thresh : int
            Threshold value.
        maxval : int, default=255
        	Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV
            thresholding types.
        threshtype : {'tozero', 'tozero_inv', 'binary', 'binary_inv', 'trunc',
                     'mask', 'otsu', 'triangle'}, optional
            Thresholding type. The default is 'tozero', which means that
            everything below `thresh` will be set to zero. See doc OpenCV.

        Yields
        ------
        Frames
            Iterator that generates thresholded frames.

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
        dsize : tuple
            Destination size (width, height).
        interpolation : {'linear', 'nearest', 'area', 'cubic', 'lanczos4'},
                        optional
            Interpolation method:
            linear - a bilinear interpolation (used by default).
            nearest - a nearest-neighbor interpolation.
            area - resampling using pixel area relation. It may be a preferred
            method for image decimation, as it gives moire-free
            results. But when the image is zoomed, it is similar to
            the nearest method.
            cubic - a bicubic interpolation over 4x4 pixel neighborhood.
            lanczos4 - a Lanczos interpolation over 8x8 pixel neighborhood.

        Yields
        ------
        Frames
            Iterator that generates resized frames.

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
        fx : float
            Scale factor along the horizontal axis.
        fy : float
            Scale factor along the vertical axis.
        interpolation : {'linear', 'nearest', 'area', 'cubic', 'lanczos4'},
                        optional
            Interpolation method:
            linear - a bilinear interpolation (used by default).
            nearest - a nearest-neighbor interpolation.
            area - resampling using pixel area relation. It may be a preferred
            method for image decimation, as it gives moire-free
            results. But when the image is zoomed, it is similar to
            the nearest method.
            cubic - a bicubic interpolation over 4x4 pixel neighborhood.
            lanczos4 - a Lanczos interpolation over 8x8 pixel neighborhood.

        Yields
        ------
        Frames
            Iterator that generates resized frames.

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

    def find_contours(self, retrmode='tree', apprmethod='simple',
                      offset=(0, 0)):
        """Finds contours in frames.

        Contours can only be performed on gray frames. Use threshold or edge
        detection before applying contours for optimal results.

        Parameters
        ----------
        retrmode : str, optional
        apprmethod : str, optional
        offset : (int, int), optional

        Yields
        ------
        Generator
            Iterator that generates tuples (contours, hierarchy), with
            contours as a tuple of arrays, and hierarchy as an array denoting
            the parent-child relationship between contours.

        """
        retrmode = {'tree': cv.RETR_TREE,
                    'external': cv.RETR_EXTERNAL,
                    'list': cv.RETR_LIST,
                    'ccomp': cv.RETR_CCOMP,
                    'floodfill': cv.RETR_FLOODFILL}[retrmode]
        apprmethod = {'none': cv.CHAIN_APPROX_NONE,
                      'simple': cv.CHAIN_APPROX_SIMPLE,
                      'tc89_l1': cv.CHAIN_APPROX_TC89_L1,
                      'tc89_kcos': cv.CHAIN_APPROX_TC89_KCOS}[apprmethod]
        for frame in self._frames:
            yield cv.findContours(frame, mode=retrmode, method=apprmethod, 
                                  offset=offset)

    def calc_meanframe(self, dtype=None):
        if self.nchannels == 1:
            meanframe = framegray(self.frameheight, self.framewidth, value=0,
                                  dtype='float64')
        else:
            meanframe = framecolor(self.frameheight, self.framewidth, color=
                                   (0, 0, 0), dtype='float64')
        for i, frame in enumerate(self._frames):
            meanframe += frame
        meanframe /= (i+1)
        if dtype is None:
            dtype = self._dtype
        return meanframe.astype(dtype)

    # TODO Frames should have a frame rate
    # TODO Exception handling
    def show(self, framerate=25):
        """Shows frames in a video window.

        Iterates through Frames and displaying each frame in a seperate
        window. Press 'q' to quit the video before the end.

        Parameters
        ----------
        framerate : int, default=25

        Notes
        -----
        Frames iterator is (partly) empty after using 'show'.

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

    def __init__(self, nframes, height, width, color=(0, 0, 0), 
                 dtype='uint8'):
        """Creates an iterator that yields color frames.

        Parameters
        ----------
        nframes : int
            Number of frames to be produced.
        height : int
            Height of frame.
        width : int
            Width of frame.
        color : tuple of ints, optional
            Fill value of frame (r, g, b). The default (0, 0, 0) color is
            black.
        dtype : numpy dtype, default='uint8'
            Dtype of frame.

        Returns
        -------
        Iterator of numpy ndarrays

        """
        frame = framecolor(height=height, width=width, color=color,
                           dtype=dtype)
        frames = (frame.copy() for _ in range(nframes))
        super().__init__(frames=frames)


class FramesGray(Frames):
    """An iterator that yields gray frames.

    This class inherits from Frames, and hence has all its methods.

    """

    def __init__(self, nframes, height, width, value=0, dtype='uint8'):
        """Creates an iterator that yields gray frames.

        Parameters
        ----------
        nframes : int
            Number of frames to be produced.
        height : int
            Height of frame.
        width : int
            Width of frame.
        value : int, optional
            Fill value of frame. The default (0) is black.
        dtype : numpy dtype, default='uint8'
            Dtype of frame.

        Returns
        -------
        Iterator of numpy ndarrays

        """

        frame = framegray(height=height, width=width, value=value,
                          dtype=dtype)
        frames = (frame.copy() for _ in range(nframes))
        super().__init__(frames=frames)


class FramesNoise(Frames):
    """An iterator that yields noise frames.

    This class inherits from Frames, and hence has all its methods.

    """

    def __init__(self, nframes, height, width, dtype='uint8'):
        """Creates an iterator that yields gray frames.

        Parameters
        ----------
        nframes : int
            Number of frames to be produced.
        height : int
            Height of frame.
        width : int
            Width of frame.
        dtype : numpy dtype, default='uint8'
            Dtype of frame.

        Returns
        -------
        Iterator of numpy ndarrays

        """

        frames = (framenoise(height=height, width=width, dtype=dtype)
                  for _ in range(nframes))
        super().__init__(frames=frames)


def framegray(height, width, value=0, dtype='uint8'):
    """Creates a gray frame.

    Parameters
    ----------
    height : int
        Height of frame.
    width : int
        Width of frame.
    value : int, optional
        Fill value of frame. The default (0) is black.
    dtype : numpy dtype, default='uint8'
        Dtype of frame.

    Returns
    -------
    numpy ndarray

    """
    return np.ones((height, width), dtype=dtype) * value


def framecolor(height, width, color=(0, 0, 0), dtype='uint8'):
    """Creates a color frame.

    Parameters
    ----------
    height : int
        Height of frame.
    width : int
        Width of frame.
    color : tuple of ints, optional
        Fill value of frame (r, g, b). The default (0, 0, 0) color is black.
    dtype : numpy dtype, default='uint8'
        Dtype of frame.

    Returns
    -------
    numpy ndarray

    """
    return np.ones((height, width, 3), dtype=dtype) * np.asanyarray(color, dtype=dtype)


def framenoise(height, width, dtype='uint8'):
    """Creates a noise frame.

    Parameters
    ----------
    height : int
        Height of frame.
    width : int
        Width of frame.
    dtype : numpy dtype, default='uint8'
        Dtype of frame.

    Returns
    -------
    numpy ndarray

    """
    return np.random.randint(0, 255, (height, width, 3), dtype=dtype)


def create_frameswithmovingcircle(nframes, width, height,
                                  framecolor=(0, 0, 0),
                                  circlecolor=(255, 100, 0), radius=6,
                                  thickness=2, linetype=8, dtype='uint8'):

    frames = FramesColor(nframes=nframes,  width=width, height=height,
                         color=framecolor, dtype=dtype)
    centers = zip(np.linspace(0, width, nframes),
                  np.linspace(0, height, nframes))
    return frames.draw_circles(centers, color=circlecolor, radius=radius,
                               thickness=thickness, linetype=linetype)
