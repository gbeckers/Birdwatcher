import unittest
import numpy as np
from numpy.testing import assert_array_equal
from birdwatcher.frames import Frames, FramesColor, framecolor, framegray


class TestFrameColor(unittest.TestCase):

    def test_color(self):
        for frame in FramesColor(5, height=480, width=640, color=(0,0,0)):
            self.assertEqual(frame.sum(), 0)


class TestFrameIterator(unittest.TestCase):

    def setUp(self):
        self.colorframes = FramesColor(5, height=480, width=640, color=(0,0,0))

    def test_drawframenumber(self):
        for frame in self.colorframes.draw_framenumbers():
            self.assertGreater(frame.sum(), 0)

class TestPeekFrame(unittest.TestCase):

    def setUp(self):
        self.frame0 = framecolor(width=640, height=480, color=(0,0,0))
        self.frame1 = framecolor(width=640, height=480, color=(1,1,1))

    def test_peekframe(self):
        frames = Frames([self.frame0, self.frame1])
        frame0 = frames.peek_frame()
        self.assertEqual(frame0.sum(), self.frame0.sum())
        framelist = [frame for frame in frames]
        self.assertEqual(len(framelist), 2) # is the first frame still available?
        self.assertEqual(framelist[1].sum(), self.frame1.sum())


class TestFindNonZero(unittest.TestCase):

    def test_grayframe(self):
        f = framegray(width=640, height=480)
        f[1, 1] = 1
        f[1, 2] = 2
        assert_array_equal(next(Frames([f]).find_nonzero()),
                           np.array([[1, 1], [2, 1]], dtype='int32'))

    def test_colorframe(self):
        f = framecolor(width=640, height=480, color=(0, 0, 0))
        f[1, 1] = (1, 1, 1)
        f[1, 2] = (1, 0, 0)
        f[1, 3] = (254, 1, 1) # sums to 0 if uint8 sum
        assert_array_equal(next(Frames([f]).find_nonzero()),
                           np.array([[1, 1], [2, 1], [3, 1]], dtype='int32'))

