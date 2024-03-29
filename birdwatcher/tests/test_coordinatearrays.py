import os
import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from birdwatcher.coordinatearrays import create_coordarray, \
    open_archivedcoordinatedata, move_coordinatearrays, CoordinateArrays


class TestCoordinateArrays(unittest.TestCase):

    def setUp(self):
        self.tempdirname1 = tempfile.mkdtemp()
        fh, self.tempvideoname = tempfile.mkstemp()
        os.close(fh)
        fh, self.temparchivename = tempfile.mkstemp(suffix='.tar.xz')
        os.close(fh)
        metadata = {'avgframerate': 5}
        self.ca1 = create_coordarray(path=self.tempdirname1,
                                    framewidth=1080, frameheight=720,
                                    metadata=metadata, overwrite=True)
        self.ca1.iterappend([((1,2),(3,4)),((5,6),(7,8))])

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)
        if Path(self.tempvideoname).exists():
            Path(self.tempvideoname).unlink()
        if Path(self.temparchivename).exists():
            Path(self.temparchivename).unlink()

    def test_index(self):
        assert_array_equal(self.ca1[1], np.array([[5,6],[7,8]]))

    def test_get_frame(self):
        frame = self.ca1.get_frame(1)
        self.assertEqual(np.sum(frame), 2)
        self.assertEqual(frame[6,5], 1)
        self.assertEqual(frame[8,7], 1)
        self.assertTupleEqual(frame.shape, (720,1080))

    def test_iter_frames(self):
        for frame in self.ca1.iter_frames():
            self.assertEqual(np.sum(frame), 2)
            self.assertTupleEqual(frame.shape, (720, 1080))

    def test_tovideo(self):
        self.ca1.tovideo(self.tempvideoname, framerate=5)

    def test_inferframerate(self):
        self.ca1.tovideo(self.tempvideoname)

    def test_coordcount(self):
        cc = self.ca1.get_coordcount()
        assert_array_equal(cc, np.array([2,2]))

    def test_coordmean(self):
        cm = self.ca1.get_coordmean()
        assert_array_equal(cm, np.array([[2,3],[6,7]]))

    def test_coordmedian(self):
        self.ca1.iterappend([((1,2),(3,4),(8,9))])
        cmd = self.ca1.get_coordmedian()
        assert_array_equal(cmd, np.array([[2,3],[6,7],[3,4]]))

    def test_open_archived(self):
        ap = self.ca1.datadir.archive(self.temparchivename, overwrite=True)
        with open_archivedcoordinatedata(ap) as ca:
            assert_array_equal(ca[1], self.ca1[1])


class TestMoveCoordinateArrays(unittest.TestCase):

    def setUp(self):
        self.tempdirname1 = tempfile.mkdtemp()
        self.tempdirname2 = tempfile.mkdtemp()
        path = Path(self.tempdirname1)/'even.darr'
        self.ca1 = create_coordarray(path=path, framewidth=1080,
                                     frameheight=720, overwrite=True)
        self.ca1.iterappend([((1, 2), (3, 4)), ((5, 6), (7, 8))])

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)
        shutil.rmtree(self.tempdirname2)

    def test_movecoordinatearrays(self):
        move_coordinatearrays(self.tempdirname1, self.tempdirname2)
        ca2 = CoordinateArrays(Path(self.tempdirname2) / 'even.darr')






