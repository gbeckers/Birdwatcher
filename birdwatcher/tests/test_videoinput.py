import unittest
import birdwatcher as bw


class TestVideos(unittest.TestCase):

    def test_videoiterframes(self):
        vf = bw.testvideosmall()
        shape = (vf.shape[1], vf.shape[0], 3)
        for i,frame in enumerate(vf.iter_frames()):
            self.assertEqual(frame.shape, shape)
        self.assertEqual(i, 496) # 497 frames



