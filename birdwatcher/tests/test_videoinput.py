import unittest
import birdwatcher as bw


class TestVideos(unittest.TestCase):

    def test_testvideosmall(self):
        vf = bw.testvideosmall()
        self.assertEqual(vf.framecount, 497)

    def test_videoiterframes(self):
        vf = bw.testvideosmall()
        shape = (vf.shape[1], vf.shape[0], 3)
        for frame in vf.iter_frames():
            self.assertEqual(frame.shape, shape)


