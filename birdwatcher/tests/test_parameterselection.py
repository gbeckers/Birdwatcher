import unittest
import tempfile
import shutil
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import birdwatcher as bw
import birdwatcher.movementdetection as md


settings = {'bgs_params':  {'History': [4, 12],
                            'ComplexityReductionThreshold': [0.05],
                            'BackgroundRatio': [0.1],
                            'NMixtures': [7],
                            'VarInit': [15],
                            'VarMin': [4],
                            'VarMax': [75],
                            'VarThreshold': [10],
                            'VarThresholdGen': [9],
                            'DetectShadows': [False],
                            'ShadowThreshold': [0.5],
                            'ShadowValue': [0]},
            'processing':  {'color': [True],
                            'resizebyfactor': [1],
                            'blur': [0,],
                            'morphologyex': [True]}}


class TestParameterSelection(unittest.TestCase):

    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        try:
            self.params = md.apply_all_parameters(bw.testvideosmall(), 
                                                  settings, nframes=200)
        except:
            self.tearDown()
            raise

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)

    def test_attributes(self):
        self.assertIsInstance(self.params, md.ParameterSelection)
        self.assertIsInstance(self.params.df, pd.DataFrame)
        self.assertIsInstance(self.params.vfs, bw.VideoFileStream)
        self.assertEqual(self.params.bgs_type, 
                         str(bw.BackgroundSubtractorMOG2))
        self.assertEqual(self.params.startat, None)
        self.assertEqual(self.params.nframes, 200)
        self.assertEqual(self.params.path, None)

    def test_getvideofragment(self):
        frames = self.params.get_videofragment()
        self.assertIsInstance(frames, bw.Frames)

    def test_getparams(self):
        all = self.params.get_parameters('all')
        settings_flat = {**settings['bgs_params'], **settings['processing']}
        self.assertEqual(all, settings_flat)
        multi_only = self.params.get_parameters('multi_only')
        self.assertEqual(multi_only, {'History': [4, 12]})

    def test_plotparams(self):
        g = self.params.plot_parameters(rows='History', cols=None, 
                                        default_values={'History': 4})
        self.assertIsInstance(g, sns.axisgrid.FacetGrid)
        self.assertEqual(g.axes.shape, (2,1))
        plt.close(g.figure)

    def test_drawcircles(self):
        frames, colorspecs = self.params.draw_multiple_circles({'History':
                                                                [4,12]})
        self.assertIsInstance(frames, bw.Frames)
        self.assertEqual(colorspecs.shape, (1,2))

    def test_saveparams(self):
        self.assertIsNone(self.params.path)
        self.params.save_parameters(self.tempdirname1)
        self.assertIsNotNone(self.params.path)

    def test_saveanotherparams(self):
        self.params.save_parameters(self.tempdirname1)
        self.params.save_parameters(self.tempdirname1)
        self.assertEqual(self.params.path[-1], '2')


class TestParamSelectionCount(unittest.TestCase):

    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        try:
            self.params = md.apply_all_parameters(bw.testvideosmall(), 
                                                  settings, nframes=200,
                                                  use_stats='count')
        except:
            self.tearDown()
            raise

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)

    def test_count(self):
        coordslabels = self.params.df.coords.unique()
        self.assertEqual(coordslabels.shape, (1,))
        self.assertEqual(coordslabels[0], 'count')