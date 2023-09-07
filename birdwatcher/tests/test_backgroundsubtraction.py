import unittest
import birdwatcher as bw
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
        for fg in frames.apply_backgroundsegmenter(bgs,  
                                                   roi=(10, 710, 10, 500),
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
        for fg in frames.apply_backgroundsegmenter(bgs,  
                                                   roi=(10, 710, 10, 500),
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