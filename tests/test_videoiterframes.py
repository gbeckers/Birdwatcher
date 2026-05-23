import unittest
import src.birdwatcher as bw


class TestVideoStreamIteration(unittest.TestCase):
    def test_frameshape(self):
        vfs = bw.testvideostreamsmall()
        shape = (vfs.frameheight, vfs.framewidth, 3)
        for i, frame in enumerate(vfs.iter_frames(), 1):
            self.assertEqual(frame.shape, shape)

    def test_nframes(self):
        vfs = bw.testvideostreamsmall()
        for i, _ in enumerate(vfs.iter_frames(), 1):
            pass
        self.assertEqual(i, 497)

    def test_startframeparameter(self):
        vfs = bw.testvideostreamsmall()
        for i, _ in enumerate(vfs.iter_frames(startframe=100), 1):
            pass
        self.assertEqual(i, 497 - 100)

    def test_nframesparameter(self):
        vfs = bw.testvideostreamsmall()
        for i, _ in enumerate(vfs.iter_frames(startframe=100, nframes=3), 1):
            pass
        self.assertEqual(i, 3)

    def test_startatparameter(self):
        vfs = bw.testvideostreamsmall()
        for i, _ in enumerate(vfs.iter_frames(startat="00:00:02."), 1):
            pass
        self.assertEqual(i, 497 - 2 * 25)

    def test_startframepriorityoverstartat(self):
        vfs = bw.testvideostreamsmall()
        for i, _ in enumerate(vfs.iter_frames(startframe=300, startat="00:00:02."), 1):
            pass
        self.assertEqual(i, 497 - 300)

    def test_stepsizeparameter(self):
        vfs = bw.testvideostreamsmall()
        for i, _ in enumerate(vfs.iter_frames(stepsize=2)):
            pass
        self.assertEqual(i, 497 // 2)
