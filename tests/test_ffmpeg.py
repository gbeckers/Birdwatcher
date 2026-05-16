import unittest
import src.birdwatcher.ffmpeg
import src.birdwatcher as bw

class TestFfmegVersion(unittest.TestCase):

    def test_str(self):
        v = src.birdwatcher.ffmpeg.ffmpegversion()
        self.assertIsInstance(v, str)


class TestSupportedEncodings(unittest.TestCase):

    def test_audio(self):
        ac = src.birdwatcher.ffmpeg.supported_audio_codecs()
        self.assertIsInstance(ac, set)
        self.assertIn('pcm_s16le', ac) # very common codec, should be supported


class TestExtractAudio(unittest.TestCase):

    def test_extract_audio_noaudio(self):
        vfs = bw.testvideostreamsmall()
        self.assertRaises(src.birdwatcher.ffmpeg.NoAudioStreamError,
                          vfs.extract_audio)