import os
from pathlib import Path
import numpy as np
import cv2 as cv
import darr

from .video import VideoFileStream
from .coordinatearrays import create_coordarray
from .backgroundsubtraction import BackgroundSubtractorMOG2, BackgroundSubtractorKNN, BackgroundSubtractorLSBP
from .utils import derive_filepath
from ._version import get_versions

__all__ = ['detect_movement', 'detect_movementmog2', 'detect_movementknn',
           'detect_movementlsbp', 'create_movementvideo']


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


def detect_movement(videofilepath, bgs, morphologyex=2, gray=True,
                    roi=None, nroi=None, analysispath='.', overwrite=False,
                    ignore_firstnframes=10, resultvideo=False):
    vf = VideoFileStream(videofilepath)
    if not Path(analysispath).exists():
        os.mkdir(analysispath)
    movementpath = Path(analysispath) / f'{vf.filepath.stem}_movement'
    if not movementpath.exists():
        os.makedirs(movementpath)
    metadata = {}
    metadata['backgroundsegmentclass'] = str(bgs)
    metadata['backgroundsegmentparams'] = bgs.get_params()
    metadata['morphologyex'] = morphologyex
    metadata['roi'] = roi
    metadata['nroi'] = nroi
    metadata['birdwatcherversion'] = get_versions()['version']
    if gray:
        frames = vf.iter_frames(color=False)
    else:
        frames = vf.iter_frames(color=True)
    frames = frames.apply_backgroundsegmenter(bgs, roi=roi, nroi=nroi)
    if morphologyex is not None:
        frames = frames.morphologyex(kernelsize=morphologyex)
    cd = create_coordarray(movementpath / 'coords.darr',
                               framewidth=vf.framewidth,
                               frameheight=vf.frameheight, metadata=metadata,
                               overwrite=overwrite)
    empty = np.zeros((0,2), dtype=np.uint16)
    coords = (c if i >= ignore_firstnframes else empty for i,c in enumerate(frames.find_nonzero()))
    cd.iterappend(coords)
    cc = darr.asarray(movementpath / 'coordscount.darr', cd.get_coordcount(),
                      metadata=metadata, overwrite=True)
    cm = darr.asarray(movementpath / 'coordsmean.darr', cd.get_coordmean(),
                      metadata=metadata, overwrite=True)
    if resultvideo:
        ovfilepath = Path(movementpath) / f'{ vf.filepath.stem}_movement.mp4'
        cframes = cd.iter_frames(nchannels=3, value=(0, 0, 255))
        (vf.iter_frames().add_weighted(0.7, cframes, 0.8)
         .draw_framenumbers()
         .tovideo(ovfilepath, framerate=vf.avgframerate, crf=25))
    return cd, cc, cm


def detect_movementknn(videofilepath, morphologyex=2, gray=True,
                        roi=None, nroi=None, analysispath='.',
                        ignore_firstnframes=10, overwrite=False,
                        **kwargs):
    bgs = BackgroundSubtractorKNN(**kwargs)
    cd, cc, cm = detect_movement(videofilepath=videofilepath,
                                 bgs=bgs,
                                 morphologyex=morphologyex,
                                 gray=gray,
                                 roi=roi,
                                 nroi=nroi,
                                 analysispath=analysispath,
                                 ignore_firstnframes=ignore_firstnframes,
                                 overwrite=overwrite)
    return cd, cc, cm


def detect_movementmog2(videofilepath, morphologyex=2, gray=True,
                        roi=None, nroi=None, analysispath='.',
                        ignore_firstnframes=10, overwrite=False,
                        **kwargs):
    bgs = BackgroundSubtractorMOG2(**kwargs)
    cd, cc, cm = detect_movement(videofilepath=videofilepath,
                                 bgs=bgs,
                                 morphologyex=morphologyex,
                                 gray=gray,
                                 roi=roi,
                                 nroi=nroi,
                                 analysispath=analysispath,
                                 ignore_firstnframes=ignore_firstnframes,
                                 overwrite=overwrite)
    return cd, cc, cm

def detect_movementlsbp(videofilepath, morphologyex=2, gray=True,
                        roi=None, nroi=None, analysispath='.',
                        ignore_firstnframes=10, overwrite=False,
                        **kwargs):
    bgs = BackgroundSubtractorLSBP(**kwargs)
    cd, cc, cm = detect_movement(videofilepath=videofilepath,
                                 bgs=bgs,
                                 morphologyex=morphologyex,
                                 gray=gray,
                                 roi=roi,
                                 nroi=nroi,
                                 analysispath=analysispath,
                                 ignore_firstnframes=ignore_firstnframes,
                                 overwrite=overwrite)
    return cd, cc, cm


def create_movementvideo(vf, ca, videofilepath=None, draw_mean=True,
                         draw_framenumbers=(2, 120), crf=17, scale=None):
    if videofilepath is None:
        videofilepath = derive_filepath(ca.path, 'results', suffix='.mp4')
    frames = ca.iter_frames(nchannels=3, value=(0,0,255)).add_weighted(0.8, vf.iter_frames(), 0.7)
    if draw_framenumbers is not None:
        frames = frames.draw_framenumbers(org=(2, 120))
    if draw_mean:
        centers = ca.get_coordmean()
        frames = frames.draw_circles(centers=centers, radius=6, color=(255, 100, 0), thickness=2, linetype=16, shift=0)
        # centers_lp = np.array(
        #     [np.convolve(centers[:, 0], np.ones(7) / 7, 'same'),
        #      np.convolve(centers[:, 1], np.ones(7) / 7, 'same')]).T
        # frames = frames.draw_circles(centers=centers_lp, radius=6, color=(100, 255, 0), thickness=2, linetype=16, shift=0)
    frames.tovideo(videofilepath, framerate=vf.avgframerate, crf=crf,
                   scale=scale)
    return ca