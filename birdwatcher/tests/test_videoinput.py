import unittest
import birdwatcher as bw


class TestVideos(unittest.TestCase):

    def test_videoiterframes(self):
        vf = bw.testvideosmall()
        shape = (vf.frameheight, vf.framewidth, 3)
        for i,frame in enumerate(vf.iter_frames(), 1):
            self.assertEqual(frame.shape, shape)
        self.assertEqual(i, 497)

    def test_size(self):
        vf = bw.testvideosmall()
        self.assertEqual(vf.framewidth, 1280)
        self.assertEqual(vf.frameheight, 720)
        self.assertSequenceEqual(vf.framesize, (1280, 720))

    def test_countframes(self):
        vf = bw.testvideosmall()
        self.assertEqual(vf.count_frames(), 497)



