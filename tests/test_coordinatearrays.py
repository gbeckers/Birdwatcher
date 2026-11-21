import os
import stat
import time
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal

import birdwatcher as bw
from birdwatcher.coordinatearrays import _archive


class TestCoordinateArrays(unittest.TestCase):
    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        fh, self.tempvideoname1 = tempfile.mkstemp()
        os.close(fh)
        fh, self.tempvideoname2 = tempfile.mkstemp()
        os.close(fh)
        metadata = {"avgframerate": 5}
        self.ca1 = bw.create_coordarray(
            path=self.tempdirname1,
            framewidth=1080,
            frameheight=720,
            metadata=metadata,
            overwrite=True,
        )
        self.ca1.iterappend([((1, 2), (3, 4)), ((5, 6), (7, 8))])

    def tearDown(self):
        for p in self.tempdirname1.rglob("*"):
            if not os.access(p, os.W_OK):
                os.chmod(p, stat.S_IWUSR)
                os.remove(p)
        os.chmod(self.tempdirname1, stat.S_IWUSR)
        for p in (self.tempvideoname1, self.tempvideoname2):
            if Path(p).exists():
                Path(p).unlink()

    def test_index(self):
        assert_array_equal(self.ca1[1], np.array([[5, 6], [7, 8]]))

    def test_get_frame(self):
        frame = self.ca1.get_frame(1)
        self.assertEqual(np.sum(frame), 2)
        self.assertEqual(frame[6, 5], 1)
        self.assertEqual(frame[8, 7], 1)
        self.assertTupleEqual(frame.shape, (720, 1080))

    def test_iter_frames(self):
        for frame in self.ca1.iter_frames():
            self.assertEqual(np.sum(frame), 2)
            self.assertTupleEqual(frame.shape, (720, 1080))

    def test_tovideo(self):
        self.ca1.tovideo(self.tempvideoname1, framerate=5, overwrite=True)

    def test_inferframerate(self):
        self.ca1.tovideo(self.tempvideoname2, overwrite=True)

    def test_coordcount(self):
        cc = self.ca1.get_coordcount()
        assert_array_equal(cc, np.array([2, 2]))

    def test_coordmean(self):
        cm = self.ca1.get_coordmean()
        assert_array_equal(cm, np.array([[2, 3], [6, 7]]))

    def test_coordmedian(self):
        self.ca1.iterappend([((1, 2), (3, 4), (8, 9))])
        cmd = self.ca1.get_coordmedian()
        assert_array_equal(cmd, np.array([[2, 3], [6, 7], [3, 4]]))

    def test_metadata_framesize(self):
        self.assertEqual(self.ca1.framewidth, 1080)
        self.assertEqual(self.ca1.frameheight, 720)
        self.assertEqual(self.ca1.metadata["birdwatcher_version"], bw.__version__)
        self.assertEqual(self.ca1.metadata["avgframerate"], 5)

    def test_reopen_readonly(self):
        ca2 = bw.CoordinateArrays(self.tempdirname1)
        self.assertEqual(ca2.accessmode, "r")
        self.assertEqual(ca2.framewidth, 1080)
        self.assertEqual(ca2.frameheight, 720)
        assert_array_equal(ca2[0], np.array([[1, 2], [3, 4]]))

    def test_get_frame_color_value_dtype(self):
        frame = self.ca1.get_frame(0, nchannels=3, dtype="float64", value=255)
        self.assertTupleEqual(frame.shape, (720, 1080, 3))
        self.assertEqual(frame.dtype, np.float64)
        assert_array_equal(frame[2, 1], [255, 255, 255])
        assert_array_equal(frame[4, 3], [255, 255, 255])
        self.assertEqual(np.sum(frame), 2 * 3 * 255)

    def test_get_frame_empty(self):
        self.ca1.append(np.zeros((0, 2), dtype="uint16"))
        frame = self.ca1.get_frame(2)
        self.assertTupleEqual(frame.shape, (720, 1080))
        self.assertEqual(np.sum(frame), 0)

    def test_iter_frames_range(self):
        self.ca1.iterappend([((9, 10),), ((11, 12),)])
        frames = list(self.ca1.iter_frames(startframe=1, endframe=4, stepsize=2))
        self.assertEqual(len(frames), 2)
        assert_array_equal(frames[0], self.ca1.get_frame(1))
        assert_array_equal(frames[1], self.ca1.get_frame(3))

    def test_iter_frames_color(self):
        frames = list(self.ca1.iter_frames(nchannels=3, value=255))
        self.assertEqual(len(frames), 2)
        for frame in frames:
            self.assertTupleEqual(frame.shape, (720, 1080, 3))
            self.assertEqual(np.sum(frame), 2 * 3 * 255)

    def test_tovideo_noframerate(self):
        ca2 = bw.create_coordarray(
            path=self.tempdirname1 / "noframerate.darr",
            framewidth=64,
            frameheight=48,
        )
        ca2.iterappend([((1, 2),)])
        with self.assertRaises(ValueError):
            ca2.tovideo(self.tempvideoname1, overwrite=True)

    def test_show(self):
        with mock.patch("cv2.imshow") as imshow, mock.patch(
            "cv2.waitKey", return_value=-1
        ) as waitkey, mock.patch("cv2.destroyAllWindows") as destroy:
            self.ca1.show()
        self.assertEqual(imshow.call_count, 2)
        frame = imshow.call_args_list[0].args[1]
        self.assertTupleEqual(frame.shape, (720, 1080, 3))
        self.assertEqual(frame[2, 1].tolist(), [255, 255, 255])
        # inferred framerate from metadata (5 fps -> 200 ms)
        waitkey.assert_called_with(200)
        destroy.assert_called_once()

    def test_show_noframenumbers_framerate(self):
        with mock.patch("cv2.imshow") as imshow, mock.patch(
            "cv2.waitKey", return_value=-1
        ) as waitkey, mock.patch("cv2.destroyAllWindows"):
            self.ca1.show(framerate=10, draw_framenumbers=False)
        self.assertEqual(imshow.call_count, 2)
        # without frame numbers only the coordinate pixels are set
        frame = imshow.call_args_list[0].args[1]
        self.assertEqual(np.sum(frame), 2 * 3 * 255)
        waitkey.assert_called_with(100)

    def test_show_draw_framenumbers(self):
        with mock.patch("cv2.imshow") as imshow, mock.patch(
            "cv2.waitKey", return_value=-1
        ), mock.patch("cv2.destroyAllWindows"):
            self.ca1.show()
        frame = imshow.call_args_list[0].args[1]
        self.assertGreater(np.sum(frame), 2 * 3 * 255)

    def test_show_quit(self):
        with mock.patch("cv2.imshow") as imshow, mock.patch(
            "cv2.waitKey", return_value=ord("q")
        ), mock.patch("cv2.destroyAllWindows"):
            self.ca1.show()
        self.assertEqual(imshow.call_count, 1)

    def test_show_noframerate(self):
        ca2 = bw.create_coordarray(
            path=self.tempdirname1 / "noframerate.darr",
            framewidth=64,
            frameheight=48,
        )
        ca2.iterappend([((1, 2),)])
        with self.assertRaises(ValueError):
            ca2.show()

    def test_coordcount_range(self):
        self.ca1.iterappend([((1, 2),), ((3, 4), (5, 6), (7, 8))])
        cc = self.ca1.get_coordcount(startframeno=1, endframeno=3)
        assert_array_equal(cc, np.array([2, 1]))

    def test_coordmean_range(self):
        self.ca1.iterappend([((10, 20),)])
        cm = self.ca1.get_coordmean(startframeno=1)
        assert_array_equal(cm, np.array([[6, 7], [10, 20]]))

    def test_coordmedian_range(self):
        cmd = self.ca1.get_coordmedian(startframeno=0, endframeno=1)
        assert_array_equal(cmd, np.array([[2, 3]]))

    def test_coordcount_emptyframe(self):
        self.ca1.append(np.zeros((0, 2), dtype="uint16"))
        assert_array_equal(self.ca1.get_coordcount(), np.array([2, 2, 0]))

    def test_coordmean_emptyframe(self):
        self.ca1.append(np.zeros((0, 2), dtype="uint16"))
        cm = self.ca1.get_coordmean()
        assert_array_equal(cm, np.array([[2, 3], [6, 7], [np.nan, np.nan]]))

    def test_coordmedian_emptyframe(self):
        self.ca1.append(np.zeros((0, 2), dtype="uint16"))
        cmd = self.ca1.get_coordmedian()
        assert_array_equal(cmd, np.array([[2, 3], [6, 7], [np.nan, np.nan]]))


class TestCreateCoordinateArrays(unittest.TestCase):
    def setUp(self):
        self.tempdirname = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tempdirname)

    def test_create_nometadata_nestedpath(self):
        path = self.tempdirname / "a" / "b" / "coords.darr"
        ca = bw.create_coordarray(path=path, framewidth=64, frameheight=48)
        self.assertTrue(path.exists())
        self.assertEqual(ca.accessmode, "r+")
        self.assertEqual(len(ca), 0)
        self.assertEqual(ca.dtype, np.uint16)
        self.assertTupleEqual(ca.atom, (2,))
        md = dict(ca.metadata)
        self.assertEqual(md["framewidth"], 64)
        self.assertEqual(md["frameheight"], 48)
        self.assertEqual(md["birdwatcher_version"], bw.__version__)

    def test_create_overwrite(self):
        path = self.tempdirname / "coords.darr"
        ca = bw.create_coordarray(path=path, framewidth=64, frameheight=48)
        ca.iterappend([((1, 2),)])
        ca = bw.create_coordarray(path=path, framewidth=32, frameheight=24)
        self.assertEqual(len(ca), 0)
        self.assertEqual(ca.framewidth, 32)

    def test_create_nooverwrite(self):
        path = self.tempdirname / "coords.darr"
        bw.create_coordarray(path=path, framewidth=64, frameheight=48)
        with self.assertRaises(OSError):
            bw.create_coordarray(
                path=path, framewidth=64, frameheight=48, overwrite=False
            )

    def test_delete_coordinatearray(self):
        path = self.tempdirname / "coords.darr"
        ca = bw.create_coordarray(path=path, framewidth=64, frameheight=48)
        ca.iterappend([((1, 2),)])
        bw.delete_coordinatearray(ca)
        self.assertFalse(path.exists())


class TestArchivedCoordinateArrays(unittest.TestCase):
    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        self.archivename = self.tempdirname1.parent / (
            self.tempdirname1.name + ".tar.xz"
        )
        metadata = {"avgframerate": 5}
        self.ca1 = bw.create_coordarray(
            path=self.tempdirname1,
            framewidth=1080,
            frameheight=720,
            metadata=metadata,
            overwrite=True,
        )
        self.ca1.iterappend([((1, 2), (3, 4)), ((5, 6), (7, 8))])

    def tearDown(self):
        if self.tempdirname1.exists():
            shutil.rmtree(self.tempdirname1)
        if self.archivename.exists():
            self.archivename.unlink()

    def test_archive(self):
        _archive(self.ca1)
        self.assertTrue(self.archivename.exists())
        self.assertFalse(self.tempdirname1.exists())

    def test_open_archived(self):
        coords1 = self.ca1[1]
        _archive(self.ca1)
        with bw.open_archivedcoordinatedata(self.archivename) as ca2:
            assert_array_equal(ca2[1], coords1)

    def test_extract_archived(self):
        coords1 = self.ca1[1]
        _archive(self.ca1)
        ca2 = bw.extract_archivedcoordinatedata(self.archivename)
        assert_array_equal(ca2[1], coords1)
        self.assertEqual(ca2.path, self.tempdirname1)
        self.assertEqual(ca2.framewidth, 1080)

    def test_open_archived_temppath(self):
        coords1 = self.ca1[1]
        _archive(self.ca1)
        temppath = Path(tempfile.mkdtemp())
        try:
            with bw.open_archivedcoordinatedata(
                self.archivename, temppath=temppath
            ) as ca2:
                self.assertTrue(Path(ca2.path).is_relative_to(temppath))
                assert_array_equal(ca2[1], coords1)
                self.assertEqual(ca2.frameheight, 720)
            # temporary extraction is cleaned up after closing
            self.assertListEqual(list(temppath.iterdir()), [])
        finally:
            shutil.rmtree(temppath)

    def test_open_archived_wrongsuffix(self):
        with self.assertRaises(OSError):
            with bw.open_archivedcoordinatedata(self.tempdirname1):
                pass

    def test_extract_archived_wrongsuffix(self):
        with self.assertRaises(OSError):
            bw.extract_archivedcoordinatedata(self.tempdirname1)


class TestMoveCoordinateArrays(unittest.TestCase):
    def setUp(self):
        self.tempdirname1 = tempfile.mkdtemp()
        self.tempdirname2 = tempfile.mkdtemp()
        path = Path(self.tempdirname1) / "even.darr"
        self.ca1 = bw.create_coordarray(
            path=path, framewidth=1080, frameheight=720, overwrite=True
        )
        self.ca1.iterappend([((1, 2), (3, 4)), ((5, 6), (7, 8))])

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)
        shutil.rmtree(self.tempdirname2)

    def test_movecoordinatearrays(self):
        bw.move_coordinatearrays(self.tempdirname1, self.tempdirname2)
        ca2 = bw.CoordinateArrays(Path(self.tempdirname2) / "even.darr")
        assert_array_equal(ca2[1], np.array([[5, 6], [7, 8]]))
        self.assertFalse((Path(self.tempdirname1) / "even.darr").exists())

    def test_movecoordinatearrays_hierarchy(self):
        nested = Path(self.tempdirname1) / "sub1" / "sub2"
        nested.mkdir(parents=True)
        (nested / "video.mp4").touch()  # non-darr file should stay
        ca = bw.create_coordarray(
            path=nested / "odd.darr", framewidth=64, frameheight=48
        )
        ca.iterappend([((1, 2),)])
        targetdir = Path(self.tempdirname2) / "newtarget"
        bw.move_coordinatearrays(self.tempdirname1, targetdir)
        self.assertTrue((targetdir / "even.darr").exists())
        movedpath = targetdir / "sub1" / "sub2" / "odd.darr"
        self.assertTrue(movedpath.exists())
        self.assertFalse((nested / "odd.darr").exists())
        self.assertTrue((nested / "video.mp4").exists())
        self.assertFalse((targetdir / "sub1" / "sub2" / "video.mp4").exists())
        ca2 = bw.CoordinateArrays(movedpath)
        assert_array_equal(ca2[0], np.array([[1, 2]]))
