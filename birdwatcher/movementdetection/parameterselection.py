"""This module contains objects and functions helpfull for determining which settings result in optimal movement detection.

"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import birdwatcher as bw
import birdwatcher.movementdetection as md
from birdwatcher.utils import product_dict


__all__ = ['ParameterSelection', 'apply_all_parameters', 
           'load_parameterselection', 'product_dict']


class ParameterSelection():
    """A Pandas dataframe with movement detection results of various parameter 
    settings associated with a (fragment of a) Videofilestream.

    """
    # colors in BGR
    colors = [('orange', [0, 100, 255]),
              ('blue', [255, 0, 0]),
              ('red', [0, 0, 255]),
              ('lime', [0, 255, 0]),
              ('cyan', [255, 255, 0]),
              ('magenta', [255, 0, 255])]

    def __init__(self, df, videofilepath, bgs_type, 
                 startat, nframes, roi, nroi, path=None):
        self.df = df
        self.vfs = bw.VideoFileStream(videofilepath)
        self.bgs_type = bgs_type
        self.startat = startat
        self.nframes = nframes
        self.roi = roi
        self.nroi = nroi
        self.path = path

    def get_info(self):
        return {'vfs': str(self.vfs.filepath),
                'bgs_type': self.bgs_type,
                'startat': self.startat,
                'nframes': self.nframes,
                'roi': self.roi,
                'nroi': self.nroi}

    def get_videofragment(self):
        """Returns video fragment as Frames.

        NOTE: the whole frames are returned. If a region of interest (roi or 
        nroi) is specified, this is not visible in the videofragment.

        """
        return self.vfs.iter_frames(startat=self.startat, 
                                    nframes=self.nframes)

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
        all_parameters = {k:self.df[k].unique().tolist() for k in paramkeys}

        if selection == 'all':
            return all_parameters
        elif selection == 'multi_only':
            return {k:all_parameters[k] for k in paramkeys if 
                    len(all_parameters[k])>1}
        else:
            raise Exception(f"'{selection}' is not recognized. Please "
                            "choose between 'all' and 'multi_only'.")

    def plot_parameters(self, rows, cols, default_values):
        """Returns a figure from seaborn with subplots.

        Usefull to look at different location detection results from two 
        parameters with various values.

        Parameters
        ----------
        rows : str
            One parameter tested with multiple values.
        cols : str
            A second parameter tested with multiple values.
        default_values : dict
            All parameters that are tested with multiple settings, should be 
            added to a dictionary with each parameter as key, and the default 
            as value.

        Returns
        ------
        FacedGrid
            A seaborn object managing multiple subplots.

        """
        self._check_multi_only(default_values)
        other_values = {key:default_values[key] for key in 
                        default_values.keys()-[rows, cols]}
        df_selection = self._select_data(**other_values)

        # plot with seaborn
        g = sns.relplot(x="framenumber", y="pixel", hue="coords", style=None, 
                        col=cols, row=rows, kind="line", data=df_selection, 
                        height=3, aspect=2)
        g.figure.suptitle(str(other_values), fontsize=15, x=0.51, y=1.05)

        return g

    def batch_plot_parameters(self, default_values, overwrite=False):
        """Saves multiple figures with subplots of all combinations of 
        parameters.

        The figures are saved in a folder 'figures' in the same directory as 
        the associated ParameterSelection file. Multiple rounds of plotting 
        with different default values can be easily saved in new folders with 
        a number added to the foldername.

        Parameters
        ----------
        default_values : dict
            All parameters that are tested with multiple settings, should be 
            added to a dictionary with each parameter as key, and the default 
            as value.
        overwrite : bool, default=False
            If False, an integer number (1,2,3,etc.) will be added as suffix 
            to the foldername, if the filepath already exists.

        """
        path = self._create_path(self.path, 'figures', overwrite)

        # save default values as txt file
        with open(path / 'default_values.txt', 'w') as f:
            for key, val in default_values.items():
                f.write(f"{key}: {val}\n")

        # plot and save each combination of two parameters
        settings = self.get_parameters('multi_only')
        for rows, _ in settings.items():
            for cols, _ in settings.items():
                if rows != cols:
                    filename = f'{self.vfs.filepath.stem}_{rows}_{cols}.png'
                    g = self.plot_parameters(rows, cols, default_values)
                    g.savefig(path / filename)
                    plt.close(g.figure)
        print(f"The figures are saved in {path}")

    def draw_multiple_circles(self, all_settings, radius=60, thickness=2):
        """Returns a Frames object with circles on the videofragment.

        It is possible to plot multiple circles on the videofragment to see 
        the results from different parameter settings.

        Parameters
        ----------
        settings : dict
            All parameters that are tested with multiple settings, should be 
            added to a dictionary with each parameter as key, and the value(s) 
            that should be superimposed on the video added as list.
        radius : int, default=60
            Radius of circle.
        thickness : int, default=2
            Line thickness.

        Returns
        -------
        Frames, DataFrame
            Iterator that generates frames with multiple circles, and a Pandas 
            DataFrame with the settings for each color of the circles.

        """
        self._check_stats_is_not_count()
        self._check_multi_only(all_settings)
        self._check_number_of_colorcombinations(all_settings)

        frames = self.get_videofragment().draw_framenumbers()
        colorspecs = {}
        for i, settings in enumerate(product_dict(**all_settings)):
            colorspecs[self.colors[i][0]] = settings
            df_selection = self._select_data(**settings)
            iterdata = (df_selection.set_index(['framenumber', 'coords'])
                        .loc[:, 'pixel'].unstack().values)
            frames = frames.draw_circles(iterdata, radius=radius,
                                         color=self.colors[i][1],
                                         thickness=thickness)

        return frames, pd.DataFrame(colorspecs)

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
        path = self._create_path(path, foldername, overwrite)
        self.path = str(path)

        # add extra information to saved .csv file
        json_info = json.dumps(self.get_info())
        self.df.to_csv(path / 'parameterselection.csv', index_label=json_info)

        # save parameter settings as readme file
        with open(path / 'readme.txt', 'w') as f:
            for key, val in self.get_parameters('all').items():
                f.write(f"{key}: {val}\n")


    def _create_path(self, path, foldername, overwrite):
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

    def _select_data(self, **kwargs):
        """Returns a copy with a selection of the dataframe.

        **kwargs can be a dictionary, with keys matching column names. The
        value of each key will be selected from a copy of the dataframe.

        """
        df = self.df.copy()
        for key, value in kwargs.items():
            df = df.loc[df[key]==value]

        return df

    def _check_multi_only(self, inputdict):
        """Check if all parameters that are tested with multiple settings, are 
        included in the inputdic.

        """
        multi_only = self.get_parameters('multi_only')
        if multi_only.keys() != inputdict.keys():
            raise Exception("Make sure the input dictionary contains all keys"
                            f" with multiple values: {multi_only.keys()}")

    def _check_number_of_colorcombinations(self, settings):
        """Check if the number of settings combinations does not exceed the
        number of possible colors.

        """
        n_combinations = len(list(product_dict(**settings)))
        n_colors = len(self.colors)
        if n_combinations > n_colors:
            raise Exception(
                f"The number of settings combinations is {n_combinations}, "
                f"but a maximum of {n_colors} circles can be plotted. Reduce "
                "the number of settings combinations, or add new colors to "
                "the class attribute 'colors' to be able to plot more "
                "circles.")
            
    def _check_stats_is_not_count(self):
        if self.df.coords.unique()[0] == 'count':
            raise Exception("This function cannot be executed with count "
                            "statistics.")


def apply_all_parameters(vfs, all_settings, startat=None, nframes=None, 
                         roi=None, nroi=None, use_stats='mean',
                         bgs_type=bw.BackgroundSubtractorMOG2, 
                         reportprogress=50):
    """Run movement detection with each set of parameters.

    Parameters
    ----------
    vfs : VideoFileStream
        A Birdwatcher VideoFileStream object.
    all_settings : {dict, dict}
        Dictionary with two dictionaries. One 'bgs_params' with the parameter 
        settings from the BackgroundSubtractor and another 'processing' 
        dictionary with settings for applying color, resizebyfactor, blur and 
        morphologyex manipulations.
    startat : str, optional
        If specified, start at this time point in the video file. You can use 
        two different time unit formats: sexagesimal 
        (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
    nframes  : int, optional
        Read a specified number of frames.
    roi : (int, int, int, int), optional
        Region of interest. Only look at this rectangular region. h1,
        h2, w1, w2.
    nroi : (int, int, int, int), optional
        Not region of interest. Exclude this rectangular region. h1,
        h2, w1, w2.
    use_stats : str, default='mean'
        As default the mean pixel per frame is calculated. Another option is 
        to use 'count' to calculate to number of pixels per frame.    
    bgs_type: BackgroundSubtractor
        This can be any of the BackgroundSubtractors in Birdwatcher, e.g. 
        BackgroundSubtractorMOG2, BackgroundSubtractorKNN, 
        BackgroundSubtractorLSBP.
    reportprogress: int or bool, default=50
        The input integer represents how often the progress of applying each 
        combination of settings is printed. Use False, to turn off 
        reportprogress.

    """
    if reportprogress:
        import datetime
        starttime = datetime.datetime.now()
        n = 0

    list_with_dfs = []
    all_combinations = product_dict(**all_settings['bgs_params'],
                                    **all_settings['processing'])

    for settings in all_combinations:
        frames = md.apply_settings(vfs, settings, startat, nframes, roi, nroi, 
                                   bgs_type)
        coordinates = frames.find_nonzero()
    
        # calculate mean or count statistics of coordinates
        if use_stats == 'mean':
            coordstats = np.array([c.mean(0) if c.size>0 else (np.nan, np.nan) 
                                   for c in coordinates])
            settings['coords'] = ['x', 'y']
        elif use_stats == 'count':
            coordstats = np.array([c.shape[0] for c in coordinates])
            settings['coords'] = ['count']
        else:
            raise Exception("Make sure use_stats has 'mean' or 'count' as "
                            "input value")
            
        # add results as pandas DataFrame
        columns = pd.MultiIndex.from_frame(pd.DataFrame(settings))
        df = pd.DataFrame(coordstats, columns=columns)
        list_with_dfs.append(df)

        if reportprogress:
            n += 1
            if n % reportprogress == 0:
                diff = datetime.datetime.now() - starttime 
                print(f'{n} combinations of settings applied in'
                      f' {str(diff).split(".")[0]} hours:min:sec')

    # create long-format DataFrame
    df = pd.concat(list_with_dfs, axis=1)
    df.index.name = 'framenumber'
    df = (df.stack(df.columns.names, future_stack=True)
          .reset_index()  # stack all column levels
          .rename({0: 'pixel'}, axis=1))

    return ParameterSelection(df, vfs.filepath, str(bgs_type), startat, 
                              nframes, roi, nroi)

def load_parameterselection(path):
    """Load a parameterselection.csv file.

    Parameters
    ----------
    path : str
        Name of the directory where parameterselection.csv is saved.

    Returns
    -------
    ParameterSelection

    """
    filepath = Path(path) / 'parameterselection.csv' 
    df = pd.read_csv(filepath, index_col=0, engine='python')
    info = json.loads(df.index.names[0])
    df.index.name = None

    return ParameterSelection(df, info['vfs'], info['bgs_type'], 
                              info['startat'], info['nframes'], 
                              info['roi'], info['nroi'], path)