import unittest
import numpy as np
import shutil
import tempfile

from pathlib import Path

from birdwatcher.coordinatearrays import create_coordarray, \
    open_archivedcoordinatedata



class CoordinateArrays(unittest.TestCase):

    def setUp(self):
        self.tempdirname1 = tempfile.mkdtemp()
        _, self.tempvideoname = tempfile.mkstemp()
        _, self.temparchivename = tempfile.mkstemp(suffix='.tar.xz')
        self.ca1 = create_coordarray(path=self.tempdirname1,
                                    framewidth=1080, frameheight=720,
                                    metadata=None, overwrite=True)
        self.ca1.iterappend([((1,2),(3,4)),((5,6),(7,8))])

    # def tearDown(self):
    #     shutil.rmtree(self.tempdirname1)
    #     if Path(self.tempvideoname).exists():
    #         Path(self.tempvideoname).unlink()
    #     if Path(self.temparchivename).exists():
    #         Path(self.temparchivename).unlink()

    def test_index(self):
        self.assertEqual(np.sum((self.ca1[1]-np.array(((5,6),(7,8))))**2), 0)

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

    def test_coordcount(self):
        cc = self.ca1.get_coordcount()
        self.assertEqual(sum((cc-(2,2))**2), 0)

    def test_coordmean(self):
        cm = self.ca1.get_coordmean()
        self.assertTupleEqual(tuple(tuple(c) for c in cm), ((2,3),(6,7)))

    def test_open_archived(self):
        ap = self.ca1.datadir.archive(self.temparchivename, overwrite=True)
        with open_archivedcoordinatedata(ap) as ca:
            self.assertEqual(np.sum((ca[1]-self.ca1[1])**2), 0)



