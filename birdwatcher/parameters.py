"""This module contains objects and functions helpfull for determining which settings result in optimal movement detection. 

"""

import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .video import VideoFileStream
from .backgroundsubtraction import BackgroundSubtractorMOG2, \
    BackgroundSubtractorKNN, BackgroundSubtractorLSBP


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def get_all_combinations(**kwargs):
    return list(product_dict(**kwargs))


def apply_all_parameters(vfs, params, startat=None, duration=None):
    """Run movement detection with each set of parameters.
    
    Parameters
    ----------
    vfs : VideoFileStream
        A Birdwatcher VideoFileStream object
    params : dict
        Dictionary with parameter settings from the backgroundSubtractorMOG2 
        and settings for applying color, resizebyfactor, blur and morphologyex 
        manipulations.
    startat : str, optional
        If specified, start at this time point in the video file. You can use 
        two different time unit formats: sexagesimal 
        (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
    duration : int, optional
        Duration of video fragment in seconds.
    
    
    """
    
    nframes = vfs.avgframerate*duration if duration else None
    
    list_with_dfs = []

    for settings in product_dict(**params):
        
        frames = vfs.iter_frames(startat=startat, nframes=nframes, 
                                 color=settings['color'])
        
        if settings['resizebyfactor'] != 1:
            val = settings['resizebyfactor']
            frames = frames.resizebyfactor(val,val)
        
        if settings['blur'] != 1:
            val = settings['blur']
            frames = frames.blur((val,val))
        
        # extract bgs settings and apply bgs
        bgs_params = BackgroundSubtractorMOG2().get_params()
        bgs_settings = {p:settings[p] for p in bgs_params.keys()}
        bgs = BackgroundSubtractorMOG2(**bgs_settings)
        frames = frames.apply_backgroundsegmenter(bgs, learningRate=-1)
        
        if settings['morphologyex']:
            frames = frames.morphologyex(morphtype='open', kernelsize=2)
        
        # find mean of nonzero coordinates
        coordinates = frames.find_nonzero()
        coordsmean = np.array([c.mean(0) if c.size>0 else (np.nan, np.nan) for 
                               c in coordinates])

        # save coordsmean x,y in pandas DataFrame 
        # with associated settings as column labels
        settings['coords'] = ['x', 'y']
        columns = pd.MultiIndex.from_frame(pd.DataFrame(settings))
        df = pd.DataFrame(coordsmean, columns=columns)
        list_with_dfs.append(df)
        
    df = pd.concat(list_with_dfs, axis=1)
    
    # create long-format
    df.index.name = 'framenumber'
    df = (df.stack(list(range(df.columns.nlevels)), dropna=False)
          .reset_index()  # stack all column levels
          .rename({0: 'pixel'}, axis=1))

    return df
