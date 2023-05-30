import unittest
import tempfile
import shutil
from pathlib import Path

import birdwatcher as bw
import birdwatcher.movementdetection as md


class TestDetectMovement(unittest.TestCase):

    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        self.vfs = (bw.testvideosmall()
                      .iter_frames(nframes=20)
                      .tovideo(self.tempdirname1 / 'even1.mp4', framerate=25))

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)

    def test_MOG2(self):
        bgs = bw.BackgroundSubtractorMOG2(History=2)
        md.detect_movement(self.vfs, bgs=bgs,
                           analysispath=self.tempdirname1, overwrite=True)

    def test_KNN(self):
        bgs = bw.BackgroundSubtractorKNN(History=2)
        md.detect_movement(self.vfs, bgs=bgs,
                           analysispath=self.tempdirname1, overwrite=True)

    def test_LSBP(self):
        bgs = bw.BackgroundSubtractorLSBP(History=2)
        md.detect_movement(self.vfs, bgs=bgs,
                           analysispath=self.tempdirname1,
                           overwrite=True)


class TestBatchDetectMovement(unittest.TestCase):

    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        self.vfs = (bw.testvideosmall()
                    .iter_frames(nframes=20)
                    .tovideo(self.tempdirname1 / 'even1.mp4', framerate=25))

    def tearDown(self):
         shutil.rmtree(self.tempdirname1)

    def test_batchdetection(self):
        p1 = self.vfs
        p2 = p1.iter_frames().tovideo(self.tempdirname1 / 'even2.mp4',
                                      framerate=p1.avgframerate)
        bgs = bw.BackgroundSubtractorKNN(History=2)
        md.batch_detect_movement([p1,p2], bgs=bgs,
                                 analysispath=self.tempdirname1, 
                                 overwrite=True)
