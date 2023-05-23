from pathlib import Path
import numpy as np
import darr

import birdwatcher as bw
from birdwatcher.utils import derive_filepath


__all__ = ['batch_detect_movement', 'detect_movement', 'detect_movementmog2',
           'detect_movementknn', 'detect_movementlsbp', 
           'create_movementvideo']


def _f(rar):
    rar.archive(overwrite=True)
    darr.delete_raggedarray(rar)


def batch_detect_movement(videofilepaths, bgs, nprocesses=6, morphologyex=2,
                          color=False, roi=None, nroi=None, analysispath='.',
                          overwrite=False, ignore_firstnframes=10,
                          resultvideo=False):
    """The reason for having a special batch function, instead of just
    applying functions in a loop, is that compression of coordinate results
    takes a long time and is single-threaded. We therefore do this in
    parallel. Use the `nprocesses` parameter to specify the number of cores
    devoted to this.

    """
    from multiprocessing.pool import ThreadPool

    tobearchived = []
    for i, videofilepath in enumerate(videofilepaths):
        cd, cc, cm = detect_movement(videofilepath, bgs=bgs,
                                     morphologyex=morphologyex, color=color,
                                     roi=roi, nroi=nroi,
                                     analysispath=analysispath,
                                     overwrite=overwrite,
                                     ignore_firstnframes=ignore_firstnframes,
                                     resultvideo=resultvideo)
        tobearchived.append(cd)
        if (len(tobearchived) == nprocesses) or (i == (len(videofilepaths) - 1)):
            with ThreadPool(processes=nprocesses) as pool:
                list([i for i in pool.imap_unordered(_f, tobearchived)])
            tobearchived = []


def detect_movement(videofilestream, bgs, morphologyex=2, color=False,
                    roi=None, nroi=None, analysispath='.',
                    ignore_firstnframes=10,
                    overwrite=False, resultvideo=False):
    """Detects movement based on a background subtraction algorithm.

    The backgound subtractor should be provided as a parameter.

    Parameters
    ----------
    videofilestream : VideoFileStream
        A Birdwatcher VideoFileStream object
    bgs : BaseBackgroundSubtractor
        Can be any instance of child from BaseBackgroundSubtractor.
        Currently included in Birdwatcher are BackgroundSubtractorMOG2,
        BackgroundSubtractorKNN, BackgroundSubtractorLSBP.
    morphologyex : int, default=2
        Kernel size of MorphologeEx open processing.
    color : bool, default=False
        Should detection be done on color frames (True) or on gray frames
        (False).
    roi : (int, int, int, int), optional
        Region of interest. Only look at this rectangular region. h1,
        h2, w1, w2.
    nroi : (int, int, int, int), optional
        Not region of interest. Exclude this rectangular region. h1,
        h2, w1, w2.
    analysispath : Path or str, optional
        Where to write results to. The default writes to the current working
        directory.
    ignore_firstnframes : int, default=10
        Do not provide coordinates for the first n frames. These often have
        a lot of false positives.
    overwrite : bool, default=False
        Overwrite results or not.
    resultvideo : bool, default=False
        Automatically generate a video with results, yes or no.

    Returns
    -------
    tuple of arrays (coordinates, coordinate count, coordinate mean)
        These are Darr arrays that are disk-based.

    """
    if isinstance(videofilestream, bw.VideoFileStream):
        vfs = videofilestream
    else:
        raise TypeError(f"`videofilestream` parameter not a VideoFileStream "
                        f"object ({type(videofilestream)}).")
    
    Path(analysispath).mkdir(parents=True, exist_ok=True)
    movementpath = Path(analysispath) / f'{vfs.filepath.stem}_movement'
    Path(movementpath).mkdir(parents=True, exist_ok=True)
    
    metadata = {}
    metadata['backgroundsegmentclass'] = str(bgs)
    metadata['backgroundsegmentparams'] = bgs.get_params()
    metadata['morphologyex'] = morphologyex
    metadata['roi'] = roi
    metadata['nroi'] = nroi
    metadata['birdwatcherversion'] = bw.__version__
    
    frames = (vfs.iter_frames(color=color)
              .apply_backgroundsegmenter(bgs, roi=roi, nroi=nroi))
    if morphologyex is not None:
        frames = frames.morphologyex(kernelsize=morphologyex)
    
    cd = frames.save_nonzero(movementpath / 'coords.darr',
                             metadata = metadata,
                             ignore_firstnframes = ignore_firstnframes,
                             overwrite = overwrite)    
    cc = darr.asarray(movementpath / 'coordscount.darr', cd.get_coordcount(),
                      metadata=metadata, overwrite=True)
    cm = darr.asarray(movementpath / 'coordsmean.darr', cd.get_coordmean(),
                      metadata=metadata, overwrite=True)
    
    if resultvideo:
        ovfilepath = Path(movementpath) / f'{ vfs.filepath.stem}_movement.mp4'
        cframes = cd.iter_frames(nchannels=3, value=(0, 0, 255))
        (vfs.iter_frames().add_weighted(0.7, cframes, 0.8)
         .draw_framenumbers()
         .tovideo(ovfilepath, framerate=vfs.avgframerate, crf=25))
    return cd, cc, cm


def detect_movementknn(videofilestream, morphologyex=2, color=False,
                        roi=None, nroi=None, analysispath='.',
                        ignore_firstnframes=10, overwrite=False,
                        **kwargs):
    """Detects movement based on a KNN background segmentation algorithm.

    The parameters for the algorithm should be provided as keyword arguments.
    There are, with their defaults {'History': 5, 'kNNSamples': 10,
    'NSamples': 6, 'Dist2Threshold': 500, 'DetectShadows': False,
    'ShadowThreshold': 0.5, 'ShadowValue': 127}

    Parameters
    ----------
    videofilestream : VideoFileStream
        A Birdwatcher VideoFileStream object
    morphologyex : int, default=2
        Kernel size of MorphologeEx open processing.
    color : bool, default=False
        Should detection be done on color frames (True) or on gray frames
        (False).
    roi : (int, int, int, int), optional
        Region of interest. Only look at this rectangular region. h1,
        h2, w1, w2.
    nroi : (int, int, int, int), optional
        Not region of interest. Exclude this rectangular region. h1,
        h2, w1, w2.
    analysispath : Path or str, optional
        Where to write results to. The default writes to the current working
        directory.
    ignore_firstnframes : int, default=10
        Do not provide coordinates for the first n frames. These often have
        a lot of false positives.
    overwrite : bool, default=False
        Overwrite results or not.
    **kwargs : dict or additional keyword arguments
        Parameters for the background segmentation algorithm.

    Returns
    -------
    tuple of arrays (coordinates, coordinate count, coordinate mean)
        These are Darr arrays that are disk-based.

    """
    bgs = bw.BackgroundSubtractorKNN(**kwargs)
    cd, cc, cm = detect_movement(videofilestream=videofilestream,
                                 bgs=bgs,
                                 morphologyex=morphologyex,
                                 color=color,
                                 roi=roi,
                                 nroi=nroi,
                                 analysispath=analysispath,
                                 ignore_firstnframes=ignore_firstnframes,
                                 overwrite=overwrite)
    return cd, cc, cm


def detect_movementmog2(videofilestream, morphologyex=2, color=False,
                        roi=None, nroi=None, analysispath='.',
                        ignore_firstnframes=10, overwrite=False,
                        **kwargs):
    """Detects movement based on a MOG2 background segmentation algorithm.

    The parameters for the algorithm should be provided as keyword arguments.
    There are, with their defaults {'History': 5,
    'ComplexityReductionThreshold': 0.05, 'BackgroundRatio': 0.1, 'NMixtures':
    7, 'VarInit': 15, 'VarMin': 4, 'VarMax': 75, 'VarThreshold': 10,
    'VarThresholdGen': 9, 'DetectShadows': False, 'ShadowThreshold': 0.5,
    'ShadowValue': 127}

    Parameters
    ----------
    videofilestream : VideoFileStream
        A Birdwatcher VideoFileStream object
    morphologyex : int, default=2
        Kernel size of MorphologeEx open processing.
    color : bool, default=False
        Should detection be done on color frames (True) or on gray frames
        (False).
    roi : (int, int, int, int), optional
        Region of interest. Only look at this rectangular region. h1,
        h2, w1, w2.
    nroi : (int, int, int, int), optional
        Not region of interest. Exclude this rectangular region. h1,
        h2, w1, w2.
    analysispath : Path or str, optional
        Where to write results to. The default writes to the current working
        directory.
    ignore_firstnframes : int, default=10
        Do not provide coordinates for the first n frames. These often have
        a lot of false positives.
    overwrite : bool, default=False
        Overwrite results or not.
    **kwargs : dict or additional keyword arguments
        Parameters for the background segmentation algorithm.

    Returns
    -------
    tuple of arrays (coordinates, coordinate count, coordinate mean)
        These are Darr arrays that are disk-based.

    """
    bgs = bw.BackgroundSubtractorMOG2(**kwargs)
    cd, cc, cm = detect_movement(videofilestream=videofilestream,
                                 bgs=bgs,
                                 morphologyex=morphologyex,
                                 color=color,
                                 roi=roi,
                                 nroi=nroi,
                                 analysispath=analysispath,
                                 ignore_firstnframes=ignore_firstnframes,
                                 overwrite=overwrite)
    return cd, cc, cm

def detect_movementlsbp(videofilestream, morphologyex=2, color=False,
                        roi=None, nroi=None, analysispath='.',
                        ignore_firstnframes=10, overwrite=False,
                        **kwargs):
    """Detects movement based on a LSBP background segmentation algorithm.

    The parameters for the algorithm should be provided as keyword arguments.
    There are, with their defaults {'mc': 0, 'nSamples': 20, 'LSBPRadius': 16,
    'Tlower': 2.0, 'Tupper': 32.0, 'Tinc': 1.0, 'Tdec': 0.05, 'Rscale': 10.0,
    'Rincdec': 0.005, 'noiseRemovalThresholdFacBG': 0.0004,
    'noiseRemovalThresholdFacFG': 0.0008, 'LSBPthreshold': 8, 'minCount': 2}

    Parameters
    ----------
    videofilestream : VideoFileStream
        A Birdwatcher VideoFileStream object
    morphologyex : int, default=2
        Kernel size of MorphologeEx open processing.
    color : bool, default=False
        Should detection be done on color frames (True) or on gray frames
        (False).
    roi : (int, int, int, int), optional
        Region of interest. Only look at this rectangular region. h1,
        h2, w1, w2.
    nroi : (int, int, int, int), optional
        Not region of interest. Exclude this rectangular region. h1,
        h2, w1, w2.
    analysispath : Path or str, optional
        Where to write results to. The default writes to the current working
        directory.
    ignore_firstnframes : int, default=10
        Do not provide coordinates for the first n frames. These often have
        a lot of false positives.
    overwrite : bool, default=False
        Overwrite results or not.
    **kwargs : dict or additional keyword arguments
        Parameters for the background segmentation algorithm.

    Returns
    -------
    tuple of arrays (coordinates, coordinate count, coordinate mean)
        These are Darr arrays that are disk-based.

    """
    bgs = bw.BackgroundSubtractorLSBP(**kwargs)
    cd, cc, cm = detect_movement(videofilestream=videofilestream,
                                 bgs=bgs,
                                 morphologyex=morphologyex,
                                 color=color,
                                 roi=roi,
                                 nroi=nroi,
                                 analysispath=analysispath,
                                 ignore_firstnframes=ignore_firstnframes,
                                 overwrite=overwrite)
    return cd, cc, cm


def create_movementvideo(videofilestream, coordinatearrays,
                         videofilepath=None, draw_mean=True,
                         draw_framenumbers=(2, 25), crf=17, scale=None):
    """Create a nice video from the original video with movement detection
    results superimposed.

    Parameters
    ----------
    videofilestream : VideoFileStream
        A Birdwatcher VideoFileStream object
    coordinatearrays : CoordinateArrays
        CoordinateArrays object with movement results.
    videofilepath : Path or str, optional
        Output path. If None, writes to filepath of videofilestream.
    draw_mean : bool, default=True
        Draw the mean of the coordinates per frame, or not.
    draw_framenumbers : tuple, optional
        Draw frame numbers. A tuple of ints indicates where to draw
        them. The default (2, 25) draws numbers in the top left corner.
        To remove the framenumbers use None.
    crf : int, default=17
        Quality factor output video for ffmpeg. The default 17 is high
        quality. Use 23 for good quality.
    scale : tuple, optional
        (width, height). The default (None) does not change width and height.

    Returns
    -------
    VideoFileStream
        Videofilestream object of the output video.

    """
    if isinstance(videofilestream, bw.VideoFileStream):
        vfs = videofilestream
    else:
        raise TypeError(f"`videofilestream` parameter not a VideoFileStream "
                        f"object ({type(videofilestream)}).")
    if videofilepath is None:
        videofilepath = derive_filepath(vfs.filepath, 'results',
                                        suffix='.mp4')
    frames = coordinatearrays.iter_frames(nchannels=3, value=(0, 0, 255)).add_weighted(0.8, vfs.iter_frames(), 0.7)
    if draw_framenumbers is not None:
        frames = frames.draw_framenumbers(org=draw_framenumbers)
    if draw_mean:
        centers = coordinatearrays.get_coordmean()
        frames = frames.draw_circles(centers=centers, radius=6, color=(255, 100, 0), thickness=2, linetype=16, shift=0)
        # centers_lp = np.array(
        #     [np.convolve(centers[:, 0], np.ones(7) / 7, 'same'),
        #      np.convolve(centers[:, 1], np.ones(7) / 7, 'same')]).T
        # frames = frames.draw_circles(centers=centers_lp, radius=6, color=(100, 255, 0), thickness=2, linetype=16, shift=0)
    vfs = frames.tovideo(videofilepath, framerate=vfs.avgframerate, crf=crf,
                   scale=scale)
    return vfs