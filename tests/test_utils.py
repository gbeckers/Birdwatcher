import io
import os
import tempfile
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import src.birdwatcher.utils as utils


class TestRoiToNpIndex(unittest.TestCase):
    def test_roi_to_npindex(self):
        a = np.arange(100).reshape(10, 10)
        idx = utils.roi_to_npindex((2, 5, 3, 7))
        self.assertEqual(idx, (slice(2, 5), slice(3, 7)))
        self.assertEqual(a[idx].shape, (3, 4))


class TestDatetimeString(unittest.TestCase):
    def test_datetimestring(self):
        s = utils.datetimestring()
        self.assertEqual(len(s), 14)
        self.assertTrue(s.isdigit())


class TestProductDict(unittest.TestCase):
    def test_product_dict(self):
        result = list(utils.product_dict(a=[1, 2], b=["x", "y"]))
        self.assertEqual(len(result), 4)
        self.assertIn({"a": 1, "b": "x"}, result)
        self.assertIn({"a": 2, "b": "y"}, result)


class TestTempdir(unittest.TestCase):
    def setUp(self):
        self.basedir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.basedir)

    def test_tempdir_removed(self):
        with utils.tempdir(self.basedir) as d:
            self.assertTrue(d.is_dir())
        self.assertFalse(d.exists())

    def test_tempdir_keep(self):
        with utils.tempdir(self.basedir, keep=True) as d:
            pass
        self.assertTrue(d.exists())

    def test_tempdir_report(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with utils.tempdir(self.basedir, report=True) as d:
                pass
        output = buf.getvalue()
        self.assertIn("created tempdir", output)
        self.assertIn("removed temp dir", output)
        self.assertFalse(d.exists())


class TestPeekIterable(unittest.TestCase):
    def test_peek(self):
        first, it = utils.peek_iterable([1, 2, 3])
        self.assertEqual(first, 1)
        self.assertEqual(list(it), [1, 2, 3])

    def test_peek_empty(self):
        with self.assertRaises(ValueError):
            utils.peek_iterable([])


class TestDeriveFilepath(unittest.TestCase):
    def test_default(self):
        p = utils.derive_filepath("/some/dir/video.mp4", "results")
        self.assertEqual(p, Path("/some/dir/video_results.mp4"))

    def test_suffix_and_path(self):
        p = utils.derive_filepath(
            "/some/dir/video.mp4", "results", suffix=".csv", path="/other"
        )
        self.assertEqual(p, Path("/other/video_results.csv"))


class TestProgress(unittest.TestCase):
    def test_progress(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            utils.progress(30, 60, status="busy")
        output = buf.getvalue()
        self.assertIn("50.0%", output)
        self.assertIn("busy", output)


class TestDirectories(unittest.TestCase):
    def setUp(self):
        self.basedir = tempfile.mkdtemp()
        sub = Path(self.basedir) / "sub"
        sub.mkdir()
        (Path(self.basedir) / "a.txt").touch()
        (sub / "b.txt").touch()
        (sub / "c.mp4").touch()

    def tearDown(self):
        shutil.rmtree(self.basedir)

    def test_print_dirstructure(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            utils.print_dirstructure(self.basedir)
        output = buf.getvalue()
        self.assertIn(f"{os.path.basename(self.basedir)}/", output)
        self.assertIn("    sub/", output)
        self.assertIn("    a.txt", output)
        self.assertIn("        b.txt", output)

    def test_print_dirstructure_path(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            utils.print_dirstructure(Path(self.basedir))
        self.assertIn("        c.mp4", buf.getvalue())

    def test_walk_paths_all(self):
        paths = list(utils.walk_paths(self.basedir))
        names = sorted(p.name for p in paths if p.is_file())
        self.assertEqual(names, ["a.txt", "b.txt", "c.mp4"])

    def test_walk_paths_extension(self):
        paths = list(utils.walk_paths(self.basedir, extension=".txt"))
        self.assertEqual(sorted(p.name for p in paths), ["a.txt", "b.txt"])
        paths = list(utils.walk_paths(Path(self.basedir), extension="mp4"))
        self.assertEqual([p.name for p in paths], ["c.mp4"])
