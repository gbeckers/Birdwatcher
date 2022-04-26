import unittest
import tempfile
import birdwatcher as bw
import shutil
import time
from pathlib import Path
from birdwatcher.frames import create_frameswithmovingcircle


class TestBackgroundSubtractorKNN(unittest.TestCase):

    def test_KNNdefaultinstantiation(self):
        bw.BackgroundSubtractorKNN()

    def test_KNNparams(self):
        bgs = bw.BackgroundSubtractorKNN(History=10)
        self.assertEqual(bgs.get_params()['History'], 10)

    def test_KNNapply(self):
        bgs = bw.BackgroundSubtractorKNN(History=10)
        frames = create_frameswithmovingcircle(nframes=5, width=1080,
                                               height=720)
        for fg in frames.apply_backgroundsegmenter(bgs,  roi=(10, 710, 10, 500),
                                                   nroi=(20,30,20,30)):
            pass


class TestBackgroundSubtractorMOG2(unittest.TestCase):

    def test_MOG2defaultinstantiation(self):
        bw.BackgroundSubtractorMOG2()

    def test_MOG2params(self):
        bgs = bw.BackgroundSubtractorMOG2(History=10)
        self.assertEqual(bgs.get_params()['History'], 10)


    def test_MOG2apply(self):
        bgs = bw.BackgroundSubtractorMOG2(History=10)
        frames = create_frameswithmovingcircle(nframes=5, width=1080,
                                               height=720)
        for fg in frames.apply_backgroundsegmenter(bgs,  roi=(10, 710, 10, 500),
                                                   nroi=(20, 30, 20, 30)):
            pass


class TestBackgroundSubtractorLSBP(unittest.TestCase):

    def test_LSBPdefaultinstantiation(self):
        bw.BackgroundSubtractorLSBP()

    def test_LSBPparams(self):
        bgs = bw.BackgroundSubtractorLSBP(nSamples=10)
        self.assertEqual(bgs.get_params()['nSamples'], 10)

    def test_LSBPapply(self):
        bgs = bw.BackgroundSubtractorLSBP()
        frames = create_frameswithmovingcircle(nframes=5, width=1080,
                                               height=720)
        for fg in frames.apply_backgroundsegmenter(bgs):
            pass


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
        bw.detect_movement(self.vfs, bgs=bgs,
                           analysispath=self.tempdirname1, overwrite=True)

    def test_KNN(self):
        bgs = bw.BackgroundSubtractorKNN(History=2)
        bw.detect_movement(self.vfs, bgs=bgs,
                           analysispath=self.tempdirname1, overwrite=True)

    def test_LSBP(self):
        bgs = bw.BackgroundSubtractorLSBP(History=2)
        bw.detect_movement(self.vfs, bgs=bgs,
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
        bw.batch_detect_movement([p1,p2], bgs=bgs,
                                 analysispath=self.tempdirname1, overwrite=True)

