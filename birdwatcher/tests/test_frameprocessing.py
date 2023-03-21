import unittest
import numpy as np
from numpy.testing import assert_array_equal
from birdwatcher.frames import Frames, FramesColor, framecolor, framegray


colorlist = [framecolor(width=640, height=480, color=(0,0,0)),
             framecolor(width=640, height=480, color=(1,1,1))]

graylist = [framegray(width=640, height=480, value=0),
            framegray(width=640, height=480, value=1)]


class TestFrames(unittest.TestCase):

    def test_size(self):
        frames = Frames(colorlist)
        self.assertEqual(frames.frameheight, 480)
        self.assertEqual(frames.framewidth, 640)

    def test_color(self):
        for frame in FramesColor(5, height=480, width=640, color= (0,0,0)):
            self.assertEqual(frame.sum(), 0)


class TestPeekFrame(unittest.TestCase):

    def test_peekframe(self):
        frames = Frames(colorlist)
        frame0 = frames.peek_frame()
        self.assertEqual(frame0.sum(), colorlist[0].sum())
        outputframes = [frame for frame in frames]
        self.assertEqual(len(outputframes), 2) # is the first frame still available?
        self.assertEqual(outputframes[1].sum(), colorlist[1].sum())


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


class TestFrameIterator(unittest.TestCase):

    def test_drawframenumber(self):
        frames = FramesColor(5, height=480, width=640, color=(0,0,0))
        for frame in frames.draw_framenumbers():
            self.assertGreater(frame.sum(), 0)

    def test_edgedetection(self):
        frames = Frames(graylist).edge_detection()
        self.assertIsInstance(frames, Frames)
        outputframes = [frame for frame in frames]
        self.assertEqual(len(outputframes), 2)
        self.assertTupleEqual(graylist[0].shape, outputframes[0].shape)