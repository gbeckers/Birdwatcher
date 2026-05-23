import unittest
import src.birdwatcher as bw
import src.birdwatcher.ffmpeg


class TestVideoFile(unittest.TestCase):
    class TestExtractAudio(unittest.TestCase):
        def test_extract_audio_noaudio(self):
            vf = bw.testvideostreamsmall().videofile
            self.assertRaises(
                src.birdwatcher.ffmpeg.NoAudioStreamError, vf.extract_audio
            )


class TestVideoStream(unittest.TestCase):
    def test_size(self):
        vf = bw.testvideostreamsmall()
        self.assertEqual(vf.framewidth, 1280)
        self.assertEqual(vf.frameheight, 720)
        self.assertSequenceEqual(vf.framesize, (1280, 720))

    def test_countframes(self):
        vf = bw.testvideostreamsmall()
        self.assertEqual(vf.count_frames(), 497)

    def test_nframes(self):
        vf = bw.testvideostreamsmall()
        self.assertEqual(vf.nframes, 497)

    def test_duration(self):
        vf = bw.testvideostreamsmall()
        self.assertEqual(vf.duration, float(19.880000))

    def test_getframe(self):
        vf = bw.testvideostreamsmall()
        frame = vf.get_frame(100)
        self.assertSequenceEqual(frame.shape, (720, 1280, 3))

    def test_getframeat(self):
        vf = bw.testvideostreamsmall()
        frame = vf.get_frameat("00:10.")
        self.assertSequenceEqual(frame.shape, (720, 1280, 3))
