from pathlib import Path

import numpy as np
import darr

import birdwatcher as bw
from birdwatcher.utils import derive_filepath


__all__ = ['batch_detect_movement', 'detect_movement', 'apply_settings', 
           'create_movementvideo']


default_settings = {'color': False,   # booleans only
                    'resizebyfactor': 1,   # use '1' for no change in size
                    'blur': 0,   # use '0' for no blur
                    'morphologyex': True}   # booleans only


def _f(rar):
    rar.archive(overwrite=True)
    darr.delete_raggedarray(rar)


def batch_detect_movement(vfs_list, settings, startat=None, nframes=None, 
                          roi=None, nroi=None, 
                          bgs_type=bw.BackgroundSubtractorMOG2, 
                          analysispath='.', ignore_firstnframes=10, 
                          overwrite=False, resultvideo=False, nprocesses=6):
    """The reason for having a special batch function, instead of just
    applying functions in a loop, is that compression of coordinate results
    takes a long time and is single-threaded. We therefore do this in
    parallel. Use the `nprocesses` parameter to specify the number of cores
    devoted to this.

    """
    from multiprocessing.pool import ThreadPool

    tobearchived = []
    for i, vfs in enumerate(vfs_list):
        cd, cc, cm = detect_movement(vfs, settings, startat=startat, 
                                     nframes=nframes, roi=roi, nroi=nroi, 
                                     bgs_type=bgs_type, 
                                     analysispath=analysispath, 
                                     ignore_firstnframes=ignore_firstnframes, 
                                     overwrite=overwrite,
                                     resultvideo=resultvideo)
        tobearchived.append(cd)
        if (len(tobearchived) == nprocesses) or (i == (len(vfs_list) - 1)):
            with ThreadPool(processes=nprocesses) as pool:
                list([i for i in pool.imap_unordered(_f, tobearchived)])
            tobearchived = []


def detect_movement(vfs, settings=None, startat=None, nframes=None, roi=None, 
                    nroi=None, bgs_type=bw.BackgroundSubtractorMOG2, 
                    analysispath='.', ignore_firstnframes=10, 
                    overwrite=False, resultvideo=False):
    """Detects movement based on a background subtraction algorithm.

    The background subtractor should be provided as a parameter.

    Parameters
    ----------
    vfs : VideoFileStream
        A Birdwatcher VideoFileStream object.
    settings : dict, optional
        Dictionary with parameter settings from the BackgroundSubtractor and 
        settings for applying color, resizebyfactor, blur and morphologyex 
        manipulations. If None, the default settings of the 
        BackgroundSubtractor are used on grey color frames, including 
        morphological transformation to reduce noise.
    startat : str, optional
        If specified, start at this time point in the video file. You can use 
        two different time unit formats: sexagesimal 
        (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
    nframes  : int, optional
            Read a specified number of frames.
    roi : (int, int, int, int), optional
        Region of interest. Only look at this rectangular region. h1, h2, w1, 
        w2.
    nroi : (int, int, int, int), optional
        Not region of interest. Exclude this rectangular region. h1, h2, w1, 
        w2.
    bgs_type: BackgroundSubtractor, default=bw.BackgroundSubtractorMOG2
        This can be any of the BackgroundSubtractors in Birdwatcher, e.g. 
        BackgroundSubtractorMOG2, BackgroundSubtractorKNN, 
        BackgroundSubtractorLSBP.
    analysispath : Path or str, optional
        Where to write results to. The default writes to the current working 
        directory.
    ignore_firstnframes : int, default=10
        Do not provide coordinates for the first n frames. These often have a 
        lot of false positives.
    overwrite : bool, default=False
        Overwrite results or not.
    resultvideo : bool, default=False
        Automatically generate a video with results, yes or no.

    Returns
    -------
    tuple of arrays (coordinates, coordinate count, coordinate mean)
        These are Darr arrays that are disk-based.

    """
    if not isinstance(vfs, bw.VideoFileStream):
        raise TypeError(f"`vfs` parameter not a VideoFileStream "
                        f"object ({type(videofilestream)}).")
    
    if settings is None:
        settings = bgs_type().get_params()
        settings.update(default_settings)

    movementpath = Path(analysispath) / f'movement_{vfs.filepath.stem}'
    Path(movementpath).mkdir(parents=True, exist_ok=True)

    metadata = {}
    metadata['backgroundsegmentclass'] = str(bgs_type)
    metadata['settings'] = settings
    metadata['startat'] = startat
    metadata['nframes'] = nframes
    metadata['roi'] = roi
    metadata['nroi'] = nroi
    metadata['birdwatcherversion'] = bw.__version__

    frames = apply_settings(vfs, settings, startat, nframes, roi, nroi, 
                            bgs_type)

    cd = frames.save_nonzero(Path(movementpath) / 'coords.darr',
                             metadata = metadata,
                             ignore_firstnframes = ignore_firstnframes,
                             overwrite = overwrite)
    cc = darr.asarray(Path(movementpath) / 'coordscount.darr', 
                      cd.get_coordcount(),
                      metadata=metadata, overwrite=True)
    cm = darr.asarray(Path(movementpath) / 'coordsmean.darr', 
                      cd.get_coordmean(),
                      metadata=metadata, overwrite=True)

    if resultvideo:
        create_movementvideo(vfs, cd, startat=startat, nframes=nframes,
                             videofilepath=Path(movementpath) / 
                             'movementvideo.mp4')
    return cd, cc, cm


def apply_settings(vfs, settings, startat=None, nframes=None, roi=None, 
                   nroi=None, bgs_type=bw.BackgroundSubtractorMOG2):
    """Applies movement detection based on various parameter settings.

    The background subtractor should be provided as a parameter.

    Parameters
    ----------
    vfs : VideoFileStream
        A Birdwatcher VideoFileStream object.
    settings : dict
        Dictionary with parameter settings from the BackgroundSubtractor and 
        settings for applying color, resizebyfactor, blur and morphologyex 
        manipulations.
    startat : str, optional
        If specified, start at this time point in the video file. You can use 
        two different time unit formats: sexagesimal 
        (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
    nframes  : int, optional
            Read a specified number of frames.
    roi : (int, int, int, int), optional
        Region of interest. Only look at this rectangular region. h1, h2, w1, 
        w2.
    nroi : (int, int, int, int), optional
        Not region of interest. Exclude this rectangular region. h1, h2, w1, 
        w2.
    bgs_type: BackgroundSubtractor, default=bw.BackgroundSubtractorMOG2
        This can be any of the BackgroundSubtractors in Birdwatcher, e.g. 
        BackgroundSubtractorMOG2, BackgroundSubtractorKNN, 
        BackgroundSubtractorLSBP.

    Yields
    ------
    Frames
        Iterator that generates numpy array frames (height x width x color 
        channel).

    """
    frames = vfs.iter_frames(startat=startat, nframes=nframes, 
                             color=settings['color'])
    
    if settings['resizebyfactor'] != 1:
        val = settings['resizebyfactor']
        frames = frames.resizebyfactor(val,val)

    if settings['blur']:
        val = settings['blur']
        frames = frames.blur((val,val))
    
    bgs_params = bgs_type().get_params()
    bgs_params.update((k, v) for k, v in settings.items() if k in bgs_params)
    bgs = bgs_type(**bgs_params)
    
    frames = frames.apply_backgroundsegmenter(bgs, learningRate=-1, 
                                              roi=roi, nroi=nroi)
    if settings['morphologyex']:
        frames = frames.morphologyex(morphtype='open', kernelsize=2)
        
    return frames


def create_movementvideo(vfs, coords, startat=None, nframes=None, 
                         videofilepath=None, draw_mean=True, 
                         draw_framenumbers=(2, 25), crf=17, scale=None):
    """Create a nice video from the original video with movement detection
    results superimposed.

    Parameters
    ----------
    vfs : VideoFileStream
        A Birdwatcher VideoFileStream object
    coords : CoordinateArrays
        CoordinateArrays object with movement results.
    startat : str, optional
        If specified, start at this time point in the video file. You can use 
        two different time unit formats: sexagesimal 
        (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
    nframes  : int, optional
            Read a specified number of frames.
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
    if not isinstance(vfs, bw.VideoFileStream):
        raise TypeError(f"`vfs` parameter not a VideoFileStream "
                        f"object ({type(videofilestream)}).")

    if videofilepath is None:
        videofilepath = derive_filepath(vfs.filepath, 'results',
                                        suffix='.mp4')

    videoframes = vfs.iter_frames(startat=startat, nframes=nframes)
    frames = (coords.iter_frames(nchannels=3, value=(0, 0, 255))
              .add_weighted(0.8, videoframes, 0.7))

    if draw_framenumbers is not None:
        frames = frames.draw_framenumbers(org=draw_framenumbers)
    if draw_mean:
        centers = coords.get_coordmean()
        frames = frames.draw_circles(centers=centers, radius=6, 
                                     color=(255, 100, 0), thickness=2, 
                                     linetype=16, shift=0)
        # centers_lp = np.array(
        #     [np.convolve(centers[:, 0], np.ones(7) / 7, 'same'),
        #      np.convolve(centers[:, 1], np.ones(7) / 7, 'same')]).T
        # frames = frames.draw_circles(centers=centers_lp, radius=6, color=(100, 255, 0), thickness=2, linetype=16, shift=0)
    vfs = frames.tovideo(videofilepath, framerate=vfs.avgframerate, crf=crf, 
                         scale=scale)
    return vfs