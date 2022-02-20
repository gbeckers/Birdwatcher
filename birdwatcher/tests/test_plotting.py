import unittest
from birdwatcher.frames import framecolor, framegray
from birdwatcher.plotting import imshow_frame


class TestImshowFrame(unittest.TestCase):

    def test_imshowcolor(self):
        frame = framecolor(height=480, width=640, color=(1,2,3))
        imshow_frame(frame)

    def test_imshowgray(self):
        frame = framegray(height=480, width=640)
        imshow_frame(frame)

