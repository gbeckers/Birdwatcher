import unittest
import src.birdwatcher.ffmpeg

class TestFfmegVersion(unittest.TestCase):

    def test_str(self):
        v = src.birdwatcher.ffmpeg.ffmpegversion()
        self.assertIsInstance(v, str)
