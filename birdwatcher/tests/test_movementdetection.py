import unittest
import tempfile
import shutil
from pathlib import Path

import birdwatcher as bw
import birdwatcher.movementdetection as md


settings = {'bgs_params':  {'History': 12,
                            'ComplexityReductionThreshold': 0.05,
                            'BackgroundRatio': 0.1,
                            'NMixtures': 7,
                            'VarInit': 15,
                            'VarMin': 10,
                            'VarMax': 75,
                            'VarThreshold': 70,
                            'VarThresholdGen': 9,
                            'DetectShadows': False,
                            'ShadowThreshold': 0.5,
                            'ShadowValue': 0},
            'processing':  {'color': False,
                            'resizebyfactor': 1,
                            'blur': 10,
                            'morphologyex': True}}


class TestApplySettingst(unittest.TestCase):

    def test_applysettings(self):
        vfs = bw.testvideosmall()
        settings_flat = {**settings['bgs_params'], **settings['processing']}
        frames = md.apply_settings(vfs, settings_flat)
        self.assertIsInstance(frames, bw.Frames)


class TestDetectMovement(unittest.TestCase):

    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        self.vfs = bw.testvideosmall()

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)

    def test_detectmovement(self):
        cd, _, _ = md.detect_movement(self.vfs, nframes=200,
                                      analysispath=self.tempdirname1,
                                      overwrite=True)
        self.assertIsInstance(cd, bw.CoordinateArrays)

    def test_movementsettings(self):
        cd, _, _ = md.detect_movement(self.vfs, settings=settings, 
                                      nframes=200, 
                                      analysispath=self.tempdirname1, 
                                      overwrite=True)
        self.assertEqual(cd.metadata['settings']['History'], 12)
        self.assertEqual(cd.metadata['settings']['blur'], 10)
    
    def test_exception(self):
        with self.assertRaises(TypeError):
            md.detect_movement('not_a_videofilestream_object')


class TestBatchDetectMovement(unittest.TestCase):

    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        self.vfs1 = (bw.testvideosmall()
                    .iter_frames(nframes=200)
                    .tovideo(self.tempdirname1 / 'video1.mp4', framerate=25))
        self.vfs2 = (bw.testvideosmall()
                    .iter_frames(nframes=200)
                    .tovideo(self.tempdirname1 / 'video2.mp4', framerate=25))

    def tearDown(self):
         shutil.rmtree(self.tempdirname1)

    def test_batchdetection(self):
        md.batch_detect_movement([self.vfs1,self.vfs2], nframes=200,
                                 analysispath=self.tempdirname1, 
                                 overwrite=True, archived=False)
        self.assertTrue(Path.exists(self.tempdirname1/'movement_video1/coords.darr'))
        self.assertTrue(Path.exists(self.tempdirname1/'movement_video2/coords.darr'))

    def test_batcharchive(self):
        md.batch_detect_movement([self.vfs1,self.vfs2], nframes=200,
                                 analysispath=self.tempdirname1,
                                 overwrite=True, archived=True)
        self.assertTrue(Path.exists(self.tempdirname1/'movement_video1/coords.darr.tar.xz'))
        self.assertTrue(Path.exists(self.tempdirname1/'movement_video2/coords.darr.tar.xz'))