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


def apply_all_parameters(vfs, bgs_params, other_settings):
    """Run movement detection with each set of parameters.
    
    Parameters
    ----------
    vfs : VideoFileStream
        A Birdwatcher VideoFileStream object
    bgs_params : dict
        Dictionary wit parameter settings from BackgroundSubtractorMOG2.
    settings : dict
        Dictionary contianing the following settings:
        {'color': [True, False],       # booleans only
        'resizebyfactor': [1, (2/3)],  # use '1' for no change in size
        'blur': [1, 10],               # use '1' for no blur
        'morphologyex': [True]}        # booleans only
    
    
    """
    
    list_with_dfs = []

    for settings in product_dict(**bgs_params, **other_settings):
        
        frames = vfs.iter_frames(color=settings['color'])
        
        if settings['resizebyfactor'] != 1:
            val = settings['resizebyfactor']
            frames = frames.resizebyfactor(val,val)
        
        if settings['blur'] != 1:
            val = settings['blur']
            frames = frames.blur((val,val))
        
        # extract bgs settings and apply bgs
        params = {p:settings[p] for p in bgs_params.keys()}  
        bgs = BackgroundSubtractorMOG2(**params)
        frames = frames.apply_backgroundsegmenter(bgs, learningRate=-1)
        
        if settings['morphologyex']:
            frames = frames.morphologyex(morphtype='open', kernelsize=2)
        
        # find mean of nonzero coordinates
        coordinates = frames.find_nonzero()
        coordsmean = np.array([c.mean(0) if c.size>0 else (np.nan, np.nan) for c in coordinates])

        # save coordsmean x,y in pandas DataFrame 
        # with associated settings as column labels
        settings['coords'] = ['x', 'y']
        columns = pd.MultiIndex.from_frame(pd.DataFrame(settings))
        df = pd.DataFrame(coordsmean, columns=columns)
        list_with_dfs.append(df)
        
    df = pd.concat(list_with_dfs, axis=1)
    
    # create long-format
    df.index.name = 'framenumber'
    df_long = df.stack(list(range(df.columns.nlevels)), 
                       dropna=False).reset_index()  # stack all column level
    df_long = df_long.rename({0: 'pixel'}, axis=1)
    df_long

    return df_long
