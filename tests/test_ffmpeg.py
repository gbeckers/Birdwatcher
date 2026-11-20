import io
import subprocess
import tempfile
import threading
import unittest
import warnings
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

import birdwatcher.ffmpeg
import birdwatcher as bw
from birdwatcher.ffmpeg import FFmpegError, NoAudioStreamError


def _make_av_file(path, duration=1, size="64x48", rate=10, channels=2):
    """Generate a small video file with an AAC audio stream using ffmpeg's
    lavfi test sources. The bundled zebra finch test video has no audio, so this is how
    audio functionality is tested."""
    if channels == 2:
        audio = ["-filter_complex", "[1:a][1:a]amerge=inputs=2[a]", "-map", "[a]"]
    else:
        audio = ["-map", "1:a"]
    args = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={size}:rate={rate}:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}:sample_rate=8000",
        "-map",
        "0:v",
        *audio,
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(args, check=True, capture_output=True)
    return Path(path)


def _noiseframes(nframes=10, height=48, width=64, color=True, seed=0):
    rng = np.random.default_rng(seed)
    shape = (height, width, 3) if color else (height, width)
    return [rng.integers(0, 256, size=shape, dtype=np.uint8) for _ in range(nframes)]


def _run_with_timeout(testcase, func, timeout=60):
    """Runs func in a thread and fails the test if it does not finish in
    time, so that a deadlock fails the test instead of hanging the suite."""
    result = {}

    def target():
        try:
            result["value"] = func()
        except BaseException as e:
            result["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    testcase.assertFalse(t.is_alive(), "ffmpeg call appears to be deadlocked")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _gradientframes(nframes=10, height=48, width=64, color=True):
    """Smooth, moving gradient frames. Unlike random noise, these can be
    encoded reliably by lossy encoders such as libopenh264."""
    y, x = np.mgrid[0:height, 0:width]
    frames = []
    for i in range(nframes):
        gray = ((x + y + 4 * i) * 2 % 256).astype(np.uint8)
        frames.append(np.dstack([gray, gray[::-1], gray[:, ::-1]]) if color else gray)
    return frames


class TestFfmegVersion(unittest.TestCase):
    def test_str(self):
        v = birdwatcher.ffmpeg.ffmpegversion()
        self.assertIsInstance(v, str)


class TestSupportedEncodings(unittest.TestCase):
    def test_audio(self):
        ac = birdwatcher.ffmpeg.supported_audio_codecs()
        self.assertIsInstance(ac, set)
        self.assertIn("pcm_s16le", ac)  # very common codec, should be supported
        self.assertNotIn("=", ac)

    def test_video(self):
        vc = birdwatcher.ffmpeg.supported_video_codecs()
        self.assertIsInstance(vc, set)
        self.assertIn("rawvideo", vc)
        self.assertNotIn("pcm_s16le", vc)  # audio codec should not be listed
        self.assertNotIn("=", vc)


class TestFFmpegError(unittest.TestCase):
    def test_attributes_and_message(self):
        e = FFmpegError(["ffmpeg", "-i", "x"], 1, stdout="out", stderr="  bad  \n")
        self.assertIsInstance(e, RuntimeError)
        self.assertEqual(e.cmd, ["ffmpeg", "-i", "x"])
        self.assertEqual(e.returncode, 1)
        self.assertEqual(e.stdout, "out")
        msg = str(e)
        self.assertIn("exit code 1", msg)
        self.assertIn("Command: ffmpeg -i x", msg)
        self.assertIn("stderr:\nbad", msg)

    def test_no_stderr(self):
        self.assertNotIn("stderr", str(FFmpegError(["ffmpeg"], 2)))


class TestRunHelpers(unittest.TestCase):
    def test_run_ffmpeg_fails(self):
        with self.assertRaises(FFmpegError) as cm:
            birdwatcher.ffmpeg.run_ffmpeg(["ffmpeg", "-i", "doesnotexist.mp4"])
        self.assertNotEqual(cm.exception.returncode, 0)
        self.assertTrue(cm.exception.stderr)

    def test_run_ffprobe_json_fails(self):
        with self.assertRaises(FFmpegError):
            birdwatcher.ffmpeg.videofileinfo("doesnotexist.mp4")


class TestCheckLoglevel(unittest.TestCase):
    def test_valid(self):
        for level in ("quiet", "error", "info", "trace"):
            birdwatcher.ffmpeg._check_loglevelarg(level)

    def test_invalid(self):
        self.assertRaises(ValueError, birdwatcher.ffmpeg._check_loglevelarg, "loud")


class TestVideoFileInfo(unittest.TestCase):
    def setUp(self):
        self.path = bw.testvideostreamsmall().filepath

    def test_videofileinfo(self):
        info = birdwatcher.ffmpeg.videofileinfo(self.path)
        self.assertIn("format", info)
        self.assertIn("streams", info)
        self.assertEqual(info["streams"][0]["codec_type"], "video")

    def test_count_frames(self):
        self.assertEqual(birdwatcher.ffmpeg.count_frames(self.path), 497)

    def test_frameproperties_color(self):
        fp = birdwatcher.ffmpeg._get_frameproperties(self.path, color=True)
        self.assertEqual(fp.shape, (720, 1280, 3))
        self.assertEqual(fp.size, 720 * 1280 * 3)
        self.assertEqual(fp.pix_fmt, "bgr24")

    def test_frameproperties_gray(self):
        fp = birdwatcher.ffmpeg._get_frameproperties(self.path, color=False)
        self.assertEqual(fp.shape, (720, 1280))
        self.assertEqual(fp.size, 720 * 1280)
        self.assertEqual(fp.pix_fmt, "gray")


class TestReadVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = bw.testvideostreamsmall().filepath

    def test_iterread_nframes(self):
        frames = list(birdwatcher.ffmpeg.iterread_videofile(self.path, nframes=5))
        self.assertEqual(len(frames), 5)
        for frame in frames:
            self.assertEqual(frame.shape, (720, 1280, 3))
            self.assertEqual(frame.dtype, np.uint8)

    def test_iterread_gray(self):
        frame = next(birdwatcher.ffmpeg.iterread_videofile(self.path, color=False))
        self.assertEqual(frame.shape, (720, 1280))

    def test_iterread_all(self):
        n = sum(1 for _ in birdwatcher.ffmpeg.iterread_videofile(self.path))
        self.assertEqual(n, 497)

    def test_iterread_startframe(self):
        frames = list(
            birdwatcher.ffmpeg.iterread_videofile(self.path, startframe=490)
        )
        self.assertEqual(len(frames), 7)
        expected = birdwatcher.ffmpeg.get_frame(self.path, 490)
        np.testing.assert_array_equal(frames[0], expected)

    def test_iterread_startat(self):
        frames = list(
            birdwatcher.ffmpeg.iterread_videofile(self.path, startat="19", nframes=100)
        )
        self.assertGreater(len(frames), 0)
        self.assertLess(len(frames), 100)

    def test_iterread_badloglevel(self):
        gen = birdwatcher.ffmpeg.iterread_videofile(self.path, loglevel="loud")
        self.assertRaises(ValueError, next, gen)

    def test_iterread_badstream(self):
        gen = birdwatcher.ffmpeg.iterread_videofile(self.path, streamnumber=5)
        self.assertRaises(ValueError, next, gen)

    def test_iterread_verbose_no_deadlock(self):
        # 'trace' produces far more stderr output than a pipe buffer holds;
        # it must be drained while frames are read.
        buf = io.StringIO()
        with redirect_stderr(buf):
            n = _run_with_timeout(
                self,
                lambda: sum(
                    1
                    for _ in birdwatcher.ffmpeg.iterread_videofile(
                        self.path, nframes=100, loglevel="trace"
                    )
                ),
            )
        self.assertEqual(n, 100)
        self.assertGreater(len(buf.getvalue()), 65536)

    def test_iterread_quiet_does_not_echo(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            list(birdwatcher.ffmpeg.iterread_videofile(self.path, nframes=2))
        self.assertEqual(buf.getvalue(), "")

    def test_get_frame_badstream(self):
        self.assertRaises(
            ValueError, birdwatcher.ffmpeg.get_frame, self.path, 0, streamnumber=1
        )

    def test_get_frame_verbose_echoes(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            birdwatcher.ffmpeg.get_frame(self.path, 0, loglevel="info")
        self.assertIn("Stream", buf.getvalue())

    def test_get_frame(self):
        frame = birdwatcher.ffmpeg.get_frame(self.path, 3)
        expected = list(birdwatcher.ffmpeg.iterread_videofile(self.path, nframes=4))[3]
        np.testing.assert_array_equal(frame, expected)

    def test_get_frame_gray(self):
        frame = birdwatcher.ffmpeg.get_frame(self.path, 0, color=False)
        self.assertEqual(frame.shape, (720, 1280))

    def test_get_frame_outofrange(self):
        self.assertRaises(FFmpegError, birdwatcher.ffmpeg.get_frame, self.path, 10000)

    def test_get_frameat(self):
        frame = birdwatcher.ffmpeg.get_frameat(self.path, "00:00:01")
        self.assertEqual(frame.shape, (720, 1280, 3))
        frame = birdwatcher.ffmpeg.get_frameat(self.path, "1.5", color=False)
        self.assertEqual(frame.shape, (720, 1280))


class TestArrayToVideo(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_color(self):
        path = self.tmpdir / "color.mp4"
        out = birdwatcher.ffmpeg.arraytovideo(_gradientframes(10), path, framerate=10)
        self.assertEqual(out, path)
        self.assertTrue(path.exists())
        self.assertEqual(birdwatcher.ffmpeg.count_frames(path), 10)
        fp = birdwatcher.ffmpeg._get_frameproperties(path, color=True)
        self.assertEqual(fp.shape, (48, 64, 3))

    def test_gray(self):
        path = self.tmpdir / "gray.mp4"
        birdwatcher.ffmpeg.arraytovideo(
            _gradientframes(5, color=False), path, framerate=10
        )
        self.assertEqual(birdwatcher.ffmpeg.count_frames(path), 5)

    def test_generator_input(self):
        path = self.tmpdir / "gen.mp4"
        birdwatcher.ffmpeg.arraytovideo(
            (f for f in _gradientframes(4)), str(path), framerate=10
        )
        self.assertEqual(birdwatcher.ffmpeg.count_frames(path), 4)

    def test_lossless_roundtrip(self):
        path = self.tmpdir / "lossless.mkv"
        frames = _noiseframes(5)
        birdwatcher.ffmpeg.arraytovideo(
            frames,
            path,
            framerate=10,
            vformat="matroska",
            codec="ffv1",
            pixfmt="bgr0",
            crf=None,
        )
        readframes = list(birdwatcher.ffmpeg.iterread_videofile(path))
        self.assertEqual(len(readframes), 5)
        for f1, f2 in zip(frames, readframes):
            np.testing.assert_array_equal(f1, f2)

    def test_lossless_roundtrip_gray(self):
        path = self.tmpdir / "lossless_gray.mkv"
        frames = _noiseframes(5, color=False)
        birdwatcher.ffmpeg.arraytovideo(
            frames,
            path,
            framerate=10,
            vformat="matroska",
            codec="ffv1",
            pixfmt="gray",
            crf=None,
        )
        readframes = list(birdwatcher.ffmpeg.iterread_videofile(path, color=False))
        for f1, f2 in zip(frames, readframes):
            np.testing.assert_array_equal(f1, f2)

    def test_noncontiguous_frames(self):
        path = self.tmpdir / "noncontig.mkv"
        frames = [f[:, ::-1] for f in _noiseframes(3)]
        self.assertFalse(frames[0].flags["C_CONTIGUOUS"])
        birdwatcher.ffmpeg.arraytovideo(
            frames,
            path,
            framerate=10,
            vformat="matroska",
            codec="ffv1",
            pixfmt="bgr0",
            crf=None,
        )
        readframes = list(birdwatcher.ffmpeg.iterread_videofile(path))
        np.testing.assert_array_equal(frames[0], readframes[0])

    def test_scale(self):
        path = self.tmpdir / "scaled.mp4"
        birdwatcher.ffmpeg.arraytovideo(
            _gradientframes(3), path, framerate=10, scale=(32, 24)
        )
        fp = birdwatcher.ffmpeg._get_frameproperties(path, color=True)
        self.assertEqual((fp.width, fp.height), (32, 24))

    def test_creates_parent_dirs(self):
        path = self.tmpdir / "a" / "b" / "video.mp4"
        birdwatcher.ffmpeg.arraytovideo(_gradientframes(2), path, framerate=10)
        self.assertTrue(path.exists())

    def test_overwrite(self):
        path = self.tmpdir / "video.mp4"
        birdwatcher.ffmpeg.arraytovideo(_gradientframes(2), path, framerate=10)
        self.assertRaises(
            IOError,
            birdwatcher.ffmpeg.arraytovideo,
            _gradientframes(2),
            path,
            framerate=10,
        )
        birdwatcher.ffmpeg.arraytovideo(
            _gradientframes(4), path, framerate=10, overwrite=True
        )
        self.assertEqual(birdwatcher.ffmpeg.count_frames(path), 4)

    def test_wrong_dtype(self):
        frames = [f.astype(np.float64) for f in _noiseframes(2)]
        self.assertRaises(
            TypeError,
            birdwatcher.ffmpeg.arraytovideo,
            frames,
            self.tmpdir / "v.mp4",
            framerate=10,
        )

    def test_wrong_nchannels(self):
        frames = [np.zeros((48, 64, 4), dtype=np.uint8)]
        self.assertRaises(
            ValueError,
            birdwatcher.ffmpeg.arraytovideo,
            frames,
            self.tmpdir / "v.mp4",
            framerate=10,
        )

    def test_wrong_ndim(self):
        frames = [np.zeros((48, 64, 3, 1), dtype=np.uint8)]
        self.assertRaises(
            ValueError,
            birdwatcher.ffmpeg.arraytovideo,
            frames,
            self.tmpdir / "v.mp4",
            framerate=10,
        )

    def test_bad_loglevel(self):
        self.assertRaises(
            ValueError,
            birdwatcher.ffmpeg.arraytovideo,
            _noiseframes(2),
            self.tmpdir / "v.mp4",
            framerate=10,
            loglevel="loud",
        )

    def test_ffmpeg_failure(self):
        # Enough frames that writing to ffmpeg's stdin fails once it has
        # exited; this must surface as FFmpegError, not BrokenPipeError.
        with self.assertRaises(FFmpegError) as cm:
            birdwatcher.ffmpeg.arraytovideo(
                _gradientframes(200, height=480, width=640),
                self.tmpdir / "v.mp4",
                framerate=10,
                codec="nonexistingcodec",
            )
        # error message is captured even with the default loglevel='quiet'
        self.assertIn("nonexistingcodec", cm.exception.stderr)

    def test_verbose_no_deadlock(self):
        path = self.tmpdir / "verbose.mp4"
        buf = io.StringIO()
        with redirect_stderr(buf):
            _run_with_timeout(
                self,
                lambda: birdwatcher.ffmpeg.arraytovideo(
                    _gradientframes(50), path, framerate=10, loglevel="debug"
                ),
            )
        self.assertEqual(birdwatcher.ffmpeg.count_frames(path), 50)
        self.assertGreater(len(buf.getvalue()), 65536)

    def test_quiet_does_not_echo(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            birdwatcher.ffmpeg.arraytovideo(
                _gradientframes(2), self.tmpdir / "q.mp4", framerate=10
            )
        self.assertEqual(buf.getvalue(), "")


class TestAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir.name)
        cls.avpath = _make_av_file(cls.tmpdir / "av.mp4")
        cls.noaudio = bw.testvideostreamsmall().filepath

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def setUp(self):
        self.outdir = Path(tempfile.mkdtemp(dir=self.tmpdir))

    def _audiostreams(self, path):
        info = birdwatcher.ffmpeg.videofileinfo(path)
        return [s for s in info["streams"] if s["codec_type"] == "audio"]

    def test_detect_audio_codec(self):
        self.assertEqual(birdwatcher.ffmpeg.detect_audio_codec(self.avpath), "aac")

    def test_detect_audio_codec_noaudio(self):
        self.assertIsNone(birdwatcher.ffmpeg.detect_audio_codec(self.noaudio))

    def test_detect_audio_codec_ffprobepath(self):
        self.assertRaises(
            FileNotFoundError,
            birdwatcher.ffmpeg.detect_audio_codec,
            self.avpath,
            ffprobepath="/nonexisting/ffprobe",
        )

    def test_detect_audio_codec_nostream(self):
        self.assertIsNone(
            birdwatcher.ffmpeg.detect_audio_codec(self.avpath, streamnumber=1)
        )

    def test_extract_copy_default_path(self):
        # default output goes next to input with codec-appropriate extension
        src = self.outdir / "av.mp4"
        src.write_bytes(self.avpath.read_bytes())
        out = birdwatcher.ffmpeg.extract_audio(src)
        self.assertEqual(out, self.outdir / "av.m4a")
        self.assertTrue(out.exists())
        streams = self._audiostreams(out)
        self.assertEqual(len(streams), 1)
        self.assertEqual(streams[0]["codec_name"], "aac")
        self.assertEqual(len(birdwatcher.ffmpeg.videofileinfo(out)["streams"]), 1)

    def test_extract_outputpath_without_suffix(self):
        out = birdwatcher.ffmpeg.extract_audio(
            self.avpath, outputpath=self.outdir / "sound"
        )
        self.assertEqual(out, self.outdir / "sound.m4a")
        self.assertTrue(out.exists())

    def test_extract_reencode(self):
        out = birdwatcher.ffmpeg.extract_audio(
            self.avpath, outputpath=self.outdir / "sound", codec="pcm_s16le"
        )
        self.assertEqual(out.suffix, ".wav")
        streams = self._audiostreams(out)
        self.assertEqual(streams[0]["codec_name"], "pcm_s16le")
        self.assertEqual(streams[0]["channels"], 2)

    def test_extract_channel(self):
        out = birdwatcher.ffmpeg.extract_audio(
            self.avpath,
            outputpath=self.outdir / "sound",
            codec="pcm_s16le",
            channel=1,
        )
        self.assertEqual(self._audiostreams(out)[0]["channels"], 1)

    def test_extract_suffix_mismatch_warns(self):
        outputpath = self.outdir / "sound.mka"
        with self.assertWarns(UserWarning):
            out = birdwatcher.ffmpeg.extract_audio(self.avpath, outputpath=outputpath)
        self.assertEqual(out, outputpath)
        self.assertTrue(out.exists())

    def test_extract_matching_suffix_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = birdwatcher.ffmpeg.extract_audio(
                self.avpath, outputpath=self.outdir / "sound.m4a"
            )
        self.assertTrue(out.exists())

    def test_extract_no_overwrite(self):
        outputpath = self.outdir / "sound"
        birdwatcher.ffmpeg.extract_audio(self.avpath, outputpath=outputpath)
        self.assertRaises(
            FileExistsError,
            birdwatcher.ffmpeg.extract_audio,
            self.avpath,
            outputpath=outputpath,
        )
        out = birdwatcher.ffmpeg.extract_audio(
            self.avpath, outputpath=outputpath, overwrite=True
        )
        self.assertTrue(out.exists())

    def test_extract_noaudio(self):
        self.assertRaises(
            NoAudioStreamError,
            birdwatcher.ffmpeg.extract_audio,
            self.noaudio,
            outputpath=self.outdir / "sound",
        )

    def test_extract_unsupported_codec(self):
        self.assertRaises(
            ValueError,
            birdwatcher.ffmpeg.extract_audio,
            self.avpath,
            outputpath=self.outdir / "sound",
            codec="nonexistingcodec",
        )

    def test_extract_bad_loglevel(self):
        self.assertRaises(
            ValueError,
            birdwatcher.ffmpeg.extract_audio,
            self.avpath,
            outputpath=self.outdir / "sound",
            loglevel="loud",
        )

    def test_extract_bad_streamnumber(self):
        for codec in ("copy", "pcm_s16le"):
            self.assertRaises(
                NoAudioStreamError,
                birdwatcher.ffmpeg.extract_audio,
                self.avpath,
                outputpath=self.outdir / "sound",
                codec=codec,
                streamnumber=3,
            )

    def test_extract_channel_with_copy(self):
        self.assertRaises(
            ValueError,
            birdwatcher.ffmpeg.extract_audio,
            self.avpath,
            outputpath=self.outdir / "sound",
            channel=1,
        )

    def test_extract_channel_zero(self):
        self.assertRaises(
            ValueError,
            birdwatcher.ffmpeg.extract_audio,
            self.avpath,
            outputpath=self.outdir / "sound",
            codec="pcm_s16le",
            channel=0,
        )

    def test_extract_channel_too_high(self):
        # ffmpeg itself silently produces silence for nonexisting channels
        self.assertRaises(
            ValueError,
            birdwatcher.ffmpeg.extract_audio,
            self.avpath,
            outputpath=self.outdir / "sound",
            codec="pcm_s16le",
            channel=3,
        )

    def test_extract_suffix_mismatch_names_codec(self):
        with self.assertWarnsRegex(UserWarning, r'codec \("aac"\)'):
            birdwatcher.ffmpeg.extract_audio(
                self.avpath, outputpath=self.outdir / "sound.mka"
            )

    def test_extract_unknown_codec_ext_no_warning(self):
        # codecs without a default extension should not warn about one
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = birdwatcher.ffmpeg.extract_audio(
                self.avpath,
                outputpath=self.outdir / "sound.mka",
                codec="pcm_u8",
            )
        self.assertTrue(out.exists())

    def test_extract_ffprobepath(self):
        self.assertRaises(
            FileNotFoundError,
            birdwatcher.ffmpeg.extract_audio,
            self.avpath,
            outputpath=self.outdir / "sound",
            ffprobepath=self.outdir / "noffprobe",
        )

    def test_extract_verbose_echoes(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            birdwatcher.ffmpeg.extract_audio(
                self.avpath, outputpath=self.outdir / "sound", loglevel="info"
            )
        self.assertIn("Stream", buf.getvalue())

    def test_extract_videofile_method(self):
        out = bw.VideoFile(self.avpath).extract_audio(
            outputpath=self.outdir / "sound", codec="pcm_s16le", channel=2
        )
        self.assertEqual(self._audiostreams(out)[0]["channels"], 1)
