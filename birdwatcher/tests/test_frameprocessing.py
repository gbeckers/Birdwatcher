import unittest
import tempfile
import shutil

import numpy as np
from numpy.testing import assert_array_equal

import birdwatcher as bw
from birdwatcher.frames import Frames, FramesColor, framecolor, framegray


colorlist = [framecolor(height=480, width=640, color=(0,0,0)),
             framecolor(height=480, width=640, color=(1,1,1))]

graylist = [framegray(height=480, width=640, value=0),
            framegray(height=480, width=640, value=1)]


class TestFrames(unittest.TestCase):

    def test_size(self):
        frames = Frames(colorlist)
        self.assertEqual(frames.frameheight, 480)
        self.assertEqual(frames.framewidth, 640)

    def test_color(self):
        for frame in FramesColor(5, height=480, width=640, color= (0,0,0)):
            self.assertEqual(frame.sum(), 0)
    
    def test_calcmeanframe(self):
        frames = Frames([framecolor(height=480, width=640, color=(10,10,10)),
                         framecolor(height=480, width=640, color=(30,30,30))])
        meanframe = frames.calc_meanframe()
        self.assertTupleEqual(meanframe.shape, (480, 640, 3))
        self.assertEqual(meanframe[0,0].sum(), 60)


class TestPeekFrame(unittest.TestCase):

    def test_peekframe(self):
        frames = Frames(colorlist)
        frame0 = frames.peek_frame()
        self.assertEqual(frame0.sum(), colorlist[0].sum())
        outputframes = [frame for frame in frames]
        self.assertEqual(len(outputframes), 2) # is the first frame still available?
        self.assertEqual(outputframes[1].sum(), colorlist[1].sum())


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


class TestFindNonZero(unittest.TestCase):
    
    def setUp(self):
        self.fgray = framegray(width=640, height=480)
        self.fgray[1,1] = 1
        self.fgray[1,2] = 2
        self.fcolor = framecolor(width=640, height=480, color=(0, 0, 0))
        self.fcolor[1,1] = (1, 1, 1)
        self.fcolor[1,2] = (1, 0, 0)
        self.fcolor[1,3] = (254, 1, 1) # sums to 0 if uint8 sum

    def test_grayframe(self):
        idx = Frames([self.fgray]).find_nonzero()
        assert_array_equal(next(idx), np.array([[1,1], [2,1]] ,dtype='int32'))

    def test_colorframe(self):
        idx = Frames([self.fcolor]).find_nonzero()
        assert_array_equal(next(idx), np.array([[1,1], [2,1], [3,1]], 
                                               dtype='int32'))
    def test_saveascoords(self):
        frames = Frames([self.fcolor, self.fcolor])
        tempdirname = tempfile.mkdtemp()        
        coordsarray = frames.save_nonzero(filepath=tempdirname, 
                                          metadata={'avgframerate': 5})
        self.assertIsInstance(coordsarray, bw.CoordinateArrays)
        shutil.rmtree(tempdirname)