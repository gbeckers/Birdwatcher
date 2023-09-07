import unittest
import birdwatcher as bw
import tempfile
import shutil

from pathlib import Path


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

    def test_nframes(self):
        vf = bw.testvideosmall()
        self.assertEqual(vf.nframes, 497)

    def test_duration(self):
        vf = bw.testvideosmall()
        self.assertEqual(vf.duration, float(19.880000))

    def test_getframe(self):
        vf = bw.testvideosmall()
        frame = vf.get_frame(100)
        self.assertSequenceEqual(frame.shape, (720,1280,3))
    
    def test_getframeat(self):
        vf = bw.testvideosmall()
        frame = vf.get_frameat('00:10.')
        self.assertSequenceEqual(frame.shape, (720,1280,3))

    def test_extractaudio(self):
        d = Path(tempfile.mkdtemp())
        p = d / 'even.wav'
        bw.testvideosmall().extract_audio(outputpath=p)
        shutil.rmtree(d)






