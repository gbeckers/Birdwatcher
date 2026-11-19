import unittest
import birdwatcher.ffmpeg
import birdwatcher as bw


class TestFfmegVersion(unittest.TestCase):
    def test_str(self):
        v = birdwatcher.ffmpeg.ffmpegversion()
        self.assertIsInstance(v, str)


class TestSupportedEncodings(unittest.TestCase):
    def test_audio(self):
        ac = birdwatcher.ffmpeg.supported_audio_codecs()
        self.assertIsInstance(ac, set)
        self.assertIn("pcm_s16le", ac)  # very common codec, should be supported
