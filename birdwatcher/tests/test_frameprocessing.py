import unittest
from birdwatcher.frameprocessing import FramesColor


class TestFrameColor(unittest.TestCase):

    def test_color(self):
        for frame in FramesColor(5, height=480, width=640, color=(0,0,0)):
            self.assertEqual(frame.sum(), 0)


class TestFrameIterator(unittest.TestCase):

    def setUp(self):
        self.colorframes = FramesColor(5, height=480, width=640, color=(0,0,0))

    def test_drawframenumber(self):
        for frame in self.colorframes.draw_framenumber():
            self.assertGreater(frame.sum(), 0)

