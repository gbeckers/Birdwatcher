import io
import shutil
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import birdwatcher as bw
import birdwatcher.ffmpeg
from birdwatcher.video import videofilesduration, walk_videofiles


def _write_wav(path, nframes=800, framerate=8000):
    """Write a short silent mono wav file (a file without video streams)."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(b"\x00\x00" * nframes)


class TestVideoFile(unittest.TestCase):
    def setUp(self):
        self.vfs = bw.testvideostreamsmall()
        self.vf = self.vfs.videofile

    def test_filepath(self):
        self.assertIsInstance(self.vf.filepath, Path)
        self.assertEqual(self.vf.filepath.name, "zf20s_low.mp4")
        self.assertEqual(self.vf.filepath, self.vfs.filepath)

    def test_formatinfo(self):
        self.assertIsInstance(self.vf.formatinfo, dict)
        self.assertIn("duration", self.vf.formatinfo)

    def test_duration(self):
        self.assertAlmostEqual(self.vf.duration, 19.88)

    def test_duration_missing(self):
        del self.vf._formatinfo["duration"]
        self.assertIsNone(self.vf.duration)

    def test_streams(self):
        self.assertEqual(self.vf.nstreams, 1)
        self.assertEqual(len(self.vf.streamsinfo), 1)
        self.assertIsInstance(self.vf.streamsinfo, tuple)

    def test_videostreams(self):
        self.assertEqual(self.vf.nvideostreams, 1)
        self.assertEqual(len(self.vf.videostreamsinfo), 1)
        self.assertEqual(self.vf.videostreamsinfo[0]["codec_type"], "video")

    def test_audiostreams(self):
        self.assertEqual(self.vf.naudiostreams, 0)
        self.assertEqual(self.vf.audiostreamsinfo, ())

    def test_repr(self):
        self.assertEqual(repr(self.vf), f"VideoFile('{self.vf.filepath}')")

    def test_str(self):
        s = str(self.vf)
        self.assertTrue(s.startswith(repr(self.vf)))
        self.assertIn("duration: 19.88", s)
        self.assertIn("number of video streams: 1", s)
        self.assertIn("number of audio streams: 0", s)

    def test_get_videostream(self):
        vfs = self.vf.get_videostream()
        self.assertIsInstance(vfs, bw.VideoFileStream)
        self.assertEqual(vfs.filepath, self.vf.filepath)

    def test_get_audiocodec_noaudio(self):
        self.assertIsNone(self.vf.get_audiocodec())

    def test_get_audiocodec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audio.wav"
            _write_wav(path)
            self.assertEqual(bw.VideoFile(path).get_audiocodec(), "pcm_s16le")

    def test_extract_audio_noaudio(self):
        self.assertRaises(birdwatcher.ffmpeg.NoAudioStreamError, self.vf.extract_audio)


class TestVideoStream(unittest.TestCase):
    def setUp(self):
        self.vfs = bw.testvideostreamsmall()

    def test_size(self):
        self.assertEqual(self.vfs.framewidth, 1280)
        self.assertEqual(self.vfs.frameheight, 720)
        self.assertSequenceEqual(self.vfs.framesize, (1280, 720))

    def test_countframes(self):
        self.assertEqual(self.vfs.count_frames(), 497)

    def test_nframes(self):
        self.assertEqual(self.vfs.nframes, 497)

    def test_nframes_missing(self):
        del self.vfs._streammetadata["nb_frames"]
        self.assertIsNone(self.vfs.nframes)

    def test_duration(self):
        self.assertEqual(self.vfs.duration, float(19.880000))

    def test_duration_missing(self):
        del self.vfs._streammetadata["duration"]
        self.assertIsNone(self.vfs.duration)

    def test_avgframerate(self):
        self.assertEqual(self.vfs.avgframerate, 25.0)

    def test_avgframerate_fallback_to_rframerate(self):
        del self.vfs._streammetadata["avg_frame_rate"]
        self.vfs._streammetadata["r_frame_rate"] = "30000/1001"
        self.assertAlmostEqual(self.vfs.avgframerate, 29.97, places=2)

    def test_avgframerate_missing(self):
        del self.vfs._streammetadata["avg_frame_rate"]
        del self.vfs._streammetadata["r_frame_rate"]
        self.assertIsNone(self.vfs.avgframerate)

    def test_codec(self):
        self.assertEqual(self.vfs.codec, "h264")

    def test_streammetadata(self):
        md = self.vfs.streammetadata
        self.assertEqual(md["codec_type"], "video")
        self.assertEqual(md["width"], 1280)

    def test_videofile(self):
        self.assertIsInstance(self.vfs.videofile, bw.VideoFile)
        self.assertEqual(self.vfs.videofile.filepath, self.vfs.filepath)

    def test_get_info(self):
        info = self.vfs.get_info()
        self.assertEqual(info["classname"], "VideoFileStream")
        self.assertEqual(
            info["classarguments"],
            {"filepath": str(self.vfs.filepath), "streamnumber": 0},
        )
        self.assertEqual(info["framewidth"], 1280)
        self.assertEqual(info["frameheight"], 720)
        self.assertEqual(info["streammetadata"], self.vfs.streammetadata)

    def test_repr(self):
        self.assertEqual(
            repr(self.vfs), f"VideoFileStream('{self.vfs.filepath}', stream=0)"
        )

    def test_str(self):
        s = str(self.vfs)
        self.assertTrue(s.startswith(repr(self.vfs)))
        self.assertIn("codec: h264", s)
        self.assertIn("avgframerate: 25.0", s)
        self.assertIn("duration: 19.88", s)
        self.assertIn("nb_frames: 497", s)
        self.assertIn("framesize: (1280, 720)", s)

    def test_iter(self):
        frame = next(iter(self.vfs))
        self.assertSequenceEqual(frame.shape, (720, 1280, 3))

    def test_iter_frames_gray(self):
        frame = next(iter(self.vfs.iter_frames(nframes=1, color=False)))
        self.assertSequenceEqual(frame.shape, (720, 1280))

    def test_iter_frames_reportprogress(self):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            frames = list(self.vfs.iter_frames(nframes=3, reportprogress=True))
        self.assertEqual(len(frames), 3)

    def test_getframe(self):
        frame = self.vfs.get_frame(100)
        self.assertSequenceEqual(frame.shape, (720, 1280, 3))

    def test_getframeat(self):
        frame = self.vfs.get_frameat("00:10.")
        self.assertSequenceEqual(frame.shape, (720, 1280, 3))

    def test_filenotfound(self):
        self.assertRaises(FileNotFoundError, bw.VideoFileStream, "doesnotexist.mp4")

    def test_streamnumber_out_of_range(self):
        self.assertRaises(
            ValueError, bw.VideoFileStream, self.vfs.filepath, streamnumber=1
        )

    def test_no_videostreams(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audio.wav"
            _write_wav(path)
            self.assertRaises(ValueError, bw.VideoFileStream, path)


class TestVideoFilesInDirectory(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.dirpath = Path(self._tmpdir.name)
        src = bw.testvideostreamsmall().filepath
        (self.dirpath / "sub").mkdir()
        shutil.copy(src, self.dirpath / "a.mp4")
        shutil.copy(src, self.dirpath / "sub" / "b.mp4")
        (self.dirpath / "notavideo.txt").write_text("hello")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_walk_videofiles(self):
        files = sorted(p.name for p in walk_videofiles(self.dirpath, extension="mp4"))
        self.assertEqual(files, ["a.mp4", "b.mp4"])

    def test_walk_videofiles_leading_dot(self):
        files = list(walk_videofiles(self.dirpath, extension=".mp4"))
        self.assertEqual(len(files), 2)

    def test_walk_videofiles_none_found(self):
        self.assertEqual(list(walk_videofiles(self.dirpath)), [])

    def test_videofilesduration(self):
        d = videofilesduration(self.dirpath, extension="mp4")
        self.assertAlmostEqual(d, 2 * 19.88)
