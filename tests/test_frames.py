"""Tests for Frames methods, properties and helper functions that are not
covered in test_frameprocessing.py."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

import birdwatcher as bw
from birdwatcher.frames import (
    Frames,
    FramesColor,
    FramesGray,
    FramesNoise,
    framecolor,
    framegray,
    framenoise,
    vstack_frames,
)

H, W = 48, 64


def _colorframes(values=(0, 0)):
    return [framecolor(height=H, width=W, color=(v, v, v)) for v in values]


def _grayframes(values=(0, 0)):
    return [framegray(height=H, width=W, value=v) for v in values]


class TestProperties(unittest.TestCase):
    def test_nchannels_color(self):
        self.assertEqual(Frames(_colorframes()).nchannels, 3)

    def test_nchannels_gray(self):
        self.assertEqual(Frames(_grayframes()).nchannels, 1)

    def test_dtype(self):
        self.assertEqual(Frames(_colorframes()).dtype, "uint8")
        self.assertEqual(FramesGray(2, H, W, dtype="uint16").dtype, "uint16")

    def test_get_info(self):
        info = Frames(_colorframes()).get_info()
        self.assertEqual(info["classname"], "Frames")
        self.assertEqual(info["framewidth"], W)
        self.assertEqual(info["frameheight"], H)
        self.assertIsNone(info["processingdata"])

    def test_processingdata_records_methods(self):
        frames = Frames(_colorframes()).blur((3, 3)).togray()
        pd = frames.get_info()["processingdata"]
        self.assertEqual([d["methodname"] for d in pd], ["blur", "togray"])
        self.assertEqual(pd[0]["methodargs"], ["(3, 3)"])

    def test_processingdata_branches_independent(self):
        base = Frames(_colorframes()).blur((3, 3))
        branch1 = base.togray()
        branch2 = base.crop(0, 10, 0, 10)
        names1 = [d["methodname"] for d in branch1.get_info()["processingdata"]]
        names2 = [d["methodname"] for d in branch2.get_info()["processingdata"]]
        self.assertEqual(names1, ["blur", "togray"])
        self.assertEqual(names2, ["blur", "crop"])
        self.assertEqual(len(base.get_info()["processingdata"]), 1)


class TestBlur(unittest.TestCase):
    def test_uniform_unchanged(self):
        out = list(Frames(_grayframes((7, 7))).blur((3, 3)))
        self.assertEqual(len(out), 2)
        assert_array_equal(out[0], _grayframes((7,))[0])

    def test_spreads_point(self):
        frame = framegray(H, W)
        frame[10, 10] = 90
        out = next(Frames([frame]).blur((3, 3)))
        self.assertEqual(out[10, 10], 10)
        self.assertEqual(out[9, 9], 10)
        self.assertEqual(out[12, 12], 0)


class TestDrawCircles(unittest.TestCase):
    def test_draws_circle(self):
        out = list(Frames(_colorframes()).draw_circles([(20, 30), (40, 10)], radius=5))
        self.assertEqual(len(out), 2)
        # pixel on circle boundary (x+radius, y) is drawn; center is not
        self.assertGreater(out[0][30, 25].sum(), 0)
        self.assertEqual(out[0][30, 20].sum(), 0)
        self.assertGreater(out[1][10, 45].sum(), 0)

    def test_nan_center_leaves_frame_unchanged(self):
        out = next(Frames(_colorframes((0,))).draw_circles([(np.nan, np.nan)]))
        self.assertEqual(out.sum(), 0)

    def test_readonly_frame(self):
        frame = framecolor(H, W)
        frame.flags.writeable = False
        out = next(Frames([frame]).draw_circles([(20, 20)]))
        self.assertGreater(out.sum(), 0)
        self.assertEqual(frame.sum(), 0)


class TestDrawRectangles(unittest.TestCase):
    def test_draws_rectangle(self):
        h1, h2, w1, w2 = 10, 30, 20, 50
        out = next(
            Frames(_colorframes((0,))).draw_rectangles([(h1, h2, w1, w2)], thickness=1)
        )
        # corners and edges drawn
        self.assertGreater(out[h1, w1].sum(), 0)
        self.assertGreater(out[h2, w2].sum(), 0)
        self.assertGreater(out[h1, 35].sum(), 0)
        self.assertGreater(out[20, w1].sum(), 0)
        # interior and exterior not drawn
        self.assertEqual(out[20, 35].sum(), 0)
        self.assertEqual(out[5, 5].sum(), 0)

    def test_nan_points_leave_frame_unchanged(self):
        out = next(Frames(_colorframes((0,))).draw_rectangles([(np.nan,) * 4]))
        self.assertEqual(out.sum(), 0)


class TestColorConversion(unittest.TestCase):
    def test_togray(self):
        out = list(Frames(_colorframes((0, 100))).togray())
        self.assertEqual(out[1].shape, (H, W))
        self.assertEqual(out[1][0, 0], 100)

    def test_togray_on_gray_is_noop(self):
        out = next(Frames(_grayframes((5,))).togray())
        assert_array_equal(out, _grayframes((5,))[0])

    def test_tocolor(self):
        out = list(Frames(_grayframes((0, 100))).tocolor())
        self.assertEqual(out[1].shape, (H, W, 3))
        assert_array_equal(out[1][0, 0], [100, 100, 100])

    def test_tocolor_on_color_is_noop(self):
        out = next(Frames(_colorframes((5,))).tocolor())
        assert_array_equal(out, _colorframes((5,))[0])


class TestDrawText(unittest.TestCase):
    def test_draws_text(self):
        out = list(Frames(_colorframes()).draw_text(["a", "b"]))
        self.assertEqual(len(out), 2)
        for frame in out:
            self.assertGreater(frame.sum(), 0)

    def test_stops_at_shortest(self):
        out = list(Frames(_colorframes()).draw_text(["a"]))
        self.assertEqual(len(out), 1)


class TestMorphologyEx(unittest.TestCase):
    def setUp(self):
        self.frame = framegray(H, W)
        self.frame[10, 10] = 255  # isolated pixel
        self.frame[20:30, 20:30] = 255  # 10x10 block

    def test_open_removes_isolated_pixel(self):
        out = next(Frames([self.frame]).morphologyex("open", kernelsize=2))
        self.assertEqual(out[10, 10], 0)
        self.assertEqual(out[25, 25], 255)

    def test_dilate_grows(self):
        out = next(Frames([self.frame]).morphologyex("dilate", kernelsize=3))
        self.assertGreater(np.count_nonzero(out), np.count_nonzero(self.frame))

    def test_erode_shrinks(self):
        out = next(Frames([self.frame]).morphologyex("erode", kernelsize=3))
        self.assertLess(np.count_nonzero(out), np.count_nonzero(self.frame))

    def test_invalid_morphtype(self):
        with self.assertRaises(ValueError):
            Frames([self.frame]).morphologyex("nonsense")


class TestAddWeighted(unittest.TestCase):
    def test_weighted_sum(self):
        frames1 = Frames(_colorframes((10, 20)))
        frames2 = Frames(_colorframes((30, 40)))
        out = list(frames1.add_weighted(0.5, frames2, 0.5, gamma=1))
        self.assertEqual(len(out), 2)
        self.assertTrue((out[0] == 21).all())
        self.assertTrue((out[1] == 31).all())


class TestCrop(unittest.TestCase):
    def test_crop(self):
        frame = framegray(H, W)
        frame[5, 7] = 9
        out = next(Frames([frame]).crop(5, 15, 7, 27))
        self.assertEqual(out.shape, (10, 20))
        self.assertEqual(out[0, 0], 9)


class TestAbsdiffFrame(unittest.TestCase):
    def test_absdiff(self):
        ref = framegray(H, W, value=50)
        out = list(Frames(_grayframes((20, 80))).absdiff_frame(ref))
        self.assertTrue((out[0] == 30).all())
        self.assertTrue((out[1] == 30).all())


class TestThreshold(unittest.TestCase):
    def setUp(self):
        self.frame = framegray(H, W, value=50)
        self.frame[:, W // 2 :] = 150

    def test_binary(self):
        out = next(Frames([self.frame]).threshold(100, threshtype="binary"))
        self.assertEqual(out[0, 0], 0)
        self.assertEqual(out[0, -1], 255)

    def test_tozero(self):
        out = next(Frames([self.frame]).threshold(100))
        self.assertEqual(out[0, 0], 0)
        self.assertEqual(out[0, -1], 150)

    def test_binary_inv(self):
        out = next(Frames([self.frame]).threshold(100, maxval=7, threshtype="binary_inv"))
        self.assertEqual(out[0, 0], 7)
        self.assertEqual(out[0, -1], 0)

    def test_invalid_threshtype(self):
        with self.assertRaises(ValueError):
            Frames([self.frame]).threshold(100, threshtype="nonsense")


class TestResize(unittest.TestCase):
    def test_resize(self):
        frames = Frames(_colorframes()).resize((32, 24))
        self.assertEqual(frames.framewidth, 32)
        self.assertEqual(frames.frameheight, 24)
        out = list(frames)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (24, 32, 3))

    def test_resize_interpolation(self):
        for interp in ("linear", "nearest", "area", "cubic", "lanczos4"):
            out = next(Frames(_grayframes((5,))).resize((16, 12), interpolation=interp))
            self.assertEqual(out.shape, (12, 16))

    def test_invalid_interpolation(self):
        with self.assertRaises(ValueError):
            Frames(_grayframes()).resize((16, 12), interpolation="nonsense")
        with self.assertRaises(ValueError):
            Frames(_grayframes()).resizebyfactor(0.5, 0.5, interpolation="nonsense")

    def test_resizebyfactor(self):
        out = next(Frames(_colorframes((0,))).resizebyfactor(0.5, 0.25))
        self.assertEqual(out.shape, (H // 4, W // 2, 3))


class TestFindContours(unittest.TestCase):
    def test_single_square(self):
        frame = framegray(H, W)
        frame[10:20, 30:40] = 255
        contours, hierarchy = next(Frames([frame]).find_contours(retrmode="external"))
        self.assertEqual(len(contours), 1)
        pts = contours[0].reshape(-1, 2)  # (x, y)
        self.assertEqual(pts[:, 0].min(), 30)
        self.assertEqual(pts[:, 0].max(), 39)
        self.assertEqual(pts[:, 1].min(), 10)
        self.assertEqual(pts[:, 1].max(), 19)

    def test_invalid_options(self):
        # raised at call time, before iteration
        with self.assertRaises(ValueError):
            Frames(_grayframes()).find_contours(retrmode="nonsense")
        with self.assertRaises(ValueError):
            Frames(_grayframes()).find_contours(apprmethod="nonsense")

    def test_no_contours(self):
        contours, hierarchy = next(Frames(_grayframes((0,))).find_contours())
        self.assertEqual(len(contours), 0)


class TestCalcMeanframe(unittest.TestCase):
    def test_gray(self):
        meanframe = Frames(_grayframes((10, 20))).calc_meanframe()
        self.assertEqual(meanframe.shape, (H, W))
        self.assertEqual(meanframe.dtype, np.uint8)
        self.assertTrue((meanframe == 15).all())

    def test_dtype(self):
        meanframe = Frames(_grayframes((10, 11))).calc_meanframe(dtype="float64")
        self.assertTrue((meanframe == 10.5).all())


class TestVstackFrames(unittest.TestCase):
    def test_vstack(self):
        frames = vstack_frames(Frames(_colorframes((1, 2))), Frames(_colorframes((3, 4))))
        self.assertIsInstance(frames, Frames)
        self.assertEqual(frames.frameheight, 2 * H)
        out = list(frames)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0, 0, 0], 1)
        self.assertEqual(out[0][-1, 0, 0], 3)


class TestFrameGenerators(unittest.TestCase):
    def test_framesgray(self):
        out = list(FramesGray(3, H, W, value=4))
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].shape, (H, W))
        self.assertTrue((out[0] == 4).all())

    def test_framescolor_frames_independent(self):
        out = list(FramesColor(2, H, W, color=(1, 2, 3)))
        assert_array_equal(out[0][0, 0], [1, 2, 3])
        out[0][0, 0] = 0
        assert_array_equal(out[1][0, 0], [1, 2, 3])

    def test_framesnoise(self):
        frames = FramesNoise(4, H, W)
        self.assertEqual(frames.nchannels, 3)
        out = list(frames)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0].shape, (H, W, 3))
        self.assertFalse(np.array_equal(out[0], out[1]))

    def test_framenoise_dtype_range(self):
        frame = framenoise(200, 200, dtype="uint16")
        self.assertEqual(frame.dtype, np.uint16)
        self.assertGreater(frame.max(), 255)


class TestSaveNonzero(unittest.TestCase):
    def setUp(self):
        self.tempdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_ignore_firstnframes_and_suffix(self):
        frame = framegray(H, W)
        frame[3, 4] = 1
        frames = Frames([frame.copy(), frame.copy(), frame.copy()])
        ca = frames.save_nonzero(self.tempdir / "coords", ignore_firstnframes=2)
        self.assertTrue((self.tempdir / "coords.darr").exists())
        self.assertEqual(len(ca), 3)
        self.assertEqual(len(ca[0]), 0)
        self.assertEqual(len(ca[1]), 0)
        assert_array_equal(ca[2], [[4, 3]])


class TestToVideo(unittest.TestCase):
    def setUp(self):
        self.tempdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_tovideo(self):
        path = self.tempdir / "out.mp4"
        vfs = FramesNoise(5, height=48, width=64).tovideo(path, framerate=10)
        self.assertIsInstance(vfs, bw.VideoFileStream)
        self.assertTrue(path.exists())
        self.assertEqual(vfs.framewidth, 64)
        self.assertEqual(vfs.frameheight, 48)
        self.assertEqual(len(list(vfs.iter_frames())), 5)


if __name__ == "__main__":
    unittest.main()
