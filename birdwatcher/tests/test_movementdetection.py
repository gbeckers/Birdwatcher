import unittest
import numpy as np
import birdwatcher as bw
from birdwatcher.frameprocessing import create_frameswithmovingcircle


class TestBackgroundSubtractorMOG2(unittest.TestCase):

    def test_MOG2defaultinstantiation(self):
        bw.BackgroundSubtractorMOG2()

    def test_MOG2getparams(self):
        bgs = bw.BackgroundSubtractorMOG2(History=10)
        params = bgs.get_params()
        self.assertEqual(params['History'], 10)

    def test_MOG2setparams(self):
        bgs = bw.BackgroundSubtractorMOG2(History=10)
        bgs.set_params(History=20)
        self.assertEqual(bgs.get_params()['History'], 20)

    def test_MOG2apply(self):
        bgs = bw.BackgroundSubtractorMOG2(History=10)
        frames = create_frameswithmovingcircle(nframes=5, width=1080,
                                               height=720)
        for frame in frames:
            thresh = bgs.apply(frame)

