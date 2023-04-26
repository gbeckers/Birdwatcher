"""This module contains objects and functions helpfull for determining which settings result in optimal movement detection.

"""

import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

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


def apply_all_parameters(vfs, settings, startat=None, duration=None):
    """Run movement detection with each set of parameters.
    
    Parameters
    ----------
    vfs : VideoFileStream
        A Birdwatcher VideoFileStream object
    settings : dict
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

    for setting in product_dict(**settings):
        
        frames = vfs.iter_frames(startat=startat, nframes=nframes, 
                                 color=setting['color'])
        
        if setting['resizebyfactor'] != 1:
            val = setting['resizebyfactor']
            frames = frames.resizebyfactor(val,val)
        
        if setting['blur'] != 1:
            val = setting['blur']
            frames = frames.blur((val,val))
        
        # extract bgs settings and apply bgs
        bgs_params = BackgroundSubtractorMOG2().get_params()
        bgs_settings = {p:setting[p] for p in bgs_params.keys()}
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
        setting['coords'] = ['x', 'y']
        columns = pd.MultiIndex.from_frame(pd.DataFrame(setting))
        df = pd.DataFrame(coordsmean, columns=columns)
        list_with_dfs.append(df)
        
    df = pd.concat(list_with_dfs, axis=1)
    
    # create long-format
    df.index.name = 'framenumber'
    df = (df.stack(list(range(df.columns.nlevels)), dropna=False)
          .reset_index()  # stack all column levels
          .rename({0: 'pixel'}, axis=1))
    
    return ParameterSelection(df, vfs.filepath, startat, duration)

def load_parameterselection(path):
    """Load a parameterselection.csv file.
    
    Parameters
    ----------
    path : str
        Name of the directory where parameterselection.csv is saved.
    
    Returns
    ------
    ParameterSelection
    
    """
    
    filepath = Path(path) / 'parameterselection.csv' 
    df = pd.read_csv(filepath, index_col=0, engine='python')
    info = eval(df.index.names[0])
    df.index.name = None
    
    return ParameterSelection(df, info['vfs'], info['startat'], 
                                 info['duration'], path)


class ParameterSelection():
    """A Pandas dataframe with movement detection results of various parameter 
    settings associated with a (fragment of a) Videofilestream.
    
    """
    
    def __init__(self, df, videofilepath, startat, duration, path=None):
        self.df = df
        self.vfs = VideoFileStream(videofilepath)
        self.startat = startat
        self.duration = duration
        self.path = path
        
    def get_info(self):
        return {'vfs': str(self.vfs.filepath),
                'startat': self.startat,
                'duration': self.duration}
    
    def get_videofragment(self):
        """Returns video fragment as Frames.
        
        """
        nframes = self.vfs.avgframerate*self.duration
        return self.vfs.iter_frames(startat=self.startat, nframes=nframes)
    
    def get_parameters(self, selection='multi_only'):
        """Returns the parameter settings used for movement detection.

        Parameters
        ----------
        selection : {'all', multi_only'}
            Specify which selection of parameters is returned:
            all : returns all parameters and their settings.
            multi_only : returns only parameters for which multiple values 
            have been used to run movement detection.
        
        Returns
        ------
        dict
            With parameters as keys, and each value contains a list of the 
            settings used for movement detection.

        """
        paramkeys = (set(self.df.columns) - 
                     set(['framenumber', 'pixel', 'coords']))
        
        all_parameters = {k:list(self.df[k].unique()) for k in paramkeys}
        
        if selection == 'all':
            return all_parameters
        elif selection == 'multi_only':
            return {k:all_parameters[k] for k in paramkeys if 
                    len(all_parameters[k])>1}
        else:
            raise Exception(f"'{selection}' is not recognized. Please "
                            "choose between 'all' and 'multi_only'.")

    
    def save_parameters(self, path, foldername=None, overwrite=False):
        """Save results of all parameter settings as .csv file.
        
        Often several rounds of parameter selection per videofragment will be 
        done with different parameter settings. For this, the same foldername 
        could be used, in which case a number is added automatically as suffix 
        to display the round.
        
        Parameters
        ----------
        path : str
            Path to disk-based directory that should be written to.
        foldername : str, optional
            Name of the folder the data should be written to.
        overwrite : bool, default=False
            If False, an integer number (1,2,3,etc.) will be added as suffix 
            to the foldername, if the filepath already exists.
        
        """
        if foldername is None:
            foldername = f'params_{self.vfs.filepath.stem}'
        path = self.create_path(path, foldername, overwrite)
        
        # add information header
        info = self.get_info()
        
        # save as .csv file
        self.df.to_csv(path / 'parameterselection.csv', index_label=info)
        
        # return path information
        self.path = str(path)


    def create_path(self, path, foldername, overwrite):
        """Useful for creating a path with a number added as suffix in case 
        the folder already exists.

        Parameters
        ----------
        path : str
            Path to disk-based directory that should be written to.
        foldername : str, optional
            Name of the folder the data should be written to.
        overwrite : bool, default=False
            If False, an integer number (1,2,3,etc.) will be added as suffix 
            to the foldername, if the filepath already exists. 

        """
        path = Path(path) / foldername

        if not overwrite:
            i = 1
            while path.exists():
                i += 1
                path = path.parent / f'{foldername}_{i}'

        Path(path).mkdir(parents=True, exist_ok=overwrite)

        return path