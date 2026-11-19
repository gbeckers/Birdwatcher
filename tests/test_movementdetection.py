import unittest
import tempfile
import shutil
from pathlib import Path

import birdwatcher as bw
import birdwatcher.movementdetection as md


settings = {
    "bgs_params": {
        "History": 12,
        "ComplexityReductionThreshold": 0.05,
        "BackgroundRatio": 0.1,
        "NMixtures": 7,
        "VarInit": 15,
        "VarMin": 10,
        "VarMax": 75,
        "VarThreshold": 70,
        "VarThresholdGen": 9,
        "DetectShadows": False,
        "ShadowThreshold": 0.5,
        "ShadowValue": 0,
    },
    "processing": {
        "color": False,
        "resizebyfactor": 1,
        "blur": 10,
        "morphologyex": True,
    },
}


class TestApplySettingst(unittest.TestCase):
    def test_applysettings(self):
        vfs = bw.testvideostreamsmall()
        settings_flat = {**settings["bgs_params"], **settings["processing"]}
        frames = md.apply_settings(vfs, settings_flat)
        self.assertIsInstance(frames, bw.Frames)

    def test_applysettings_resize_roi(self):
        vfs = bw.testvideostreamsmall()
        settings_flat = {**settings["bgs_params"], **settings["processing"]}
        settings_flat["resizebyfactor"] = 0.5
        frames = md.apply_settings(
            vfs, settings_flat, nframes=5, roi=(10, 100, 10, 200),
            nroi=(20, 40, 20, 40)
        )
        frame = next(iter(frames))
        self.assertEqual(frame.shape, (vfs.frameheight // 2, vfs.framewidth // 2))


class TestDetectMovement(unittest.TestCase):
    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        self.vfs = bw.testvideostreamsmall()

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)

    def test_detectmovement(self):
        cd, _, _ = md.detect_movement(
            self.vfs, nframes=200, analysispath=self.tempdirname1, overwrite=True
        )
        self.assertIsInstance(cd, bw.CoordinateArrays)

    def test_movementsettings(self):
        cd, _, _ = md.detect_movement(
            self.vfs,
            settings=settings,
            nframes=200,
            analysispath=self.tempdirname1,
            overwrite=True,
        )
        self.assertEqual(cd.metadata["settings"]["History"], 12)
        self.assertEqual(cd.metadata["settings"]["blur"], 10)

    def test_exception(self):
        with self.assertRaises(TypeError):
            md.detect_movement("not_a_videofilestream_object")

    def test_resultvideo(self):
        md.detect_movement(
            self.vfs,
            nframes=50,
            analysispath=self.tempdirname1,
            overwrite=True,
            resultvideo=True,
        )
        videopath = (
            self.tempdirname1 / f"movement_{self.vfs.filepath.stem}" / "movementvideo.mp4"
        )
        self.assertTrue(videopath.exists())


class TestCreateMovementVideo(unittest.TestCase):
    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        self.vfs = bw.testvideostreamsmall()

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)

    def test_exception(self):
        with self.assertRaises(TypeError):
            md.create_movementvideo("not_a_videofilestream_object", None)

    def test_no_mean_no_framenumbers(self):
        cd, _, _ = md.detect_movement(
            self.vfs, nframes=50, analysispath=self.tempdirname1, overwrite=True
        )
        videopath = self.tempdirname1 / "movementvideo.mp4"
        vfs = md.create_movementvideo(
            self.vfs,
            cd,
            nframes=50,
            videofilepath=videopath,
            draw_mean=False,
            draw_framenumbers=None,
        )
        self.assertIsInstance(vfs, bw.VideoFileStream)
        self.assertTrue(videopath.exists())


class TestBatchDetectMovement(unittest.TestCase):
    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        self.vfs1 = (
            bw.testvideostreamsmall()
            .iter_frames(nframes=200)
            .tovideo(self.tempdirname1 / "video1.mp4", framerate=25)
        )
        self.vfs2 = (
            bw.testvideostreamsmall()
            .iter_frames(nframes=200)
            .tovideo(self.tempdirname1 / "video2.mp4", framerate=25)
        )

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)

    def test_batchdetection(self):
        md.batch_detect_movement(
            [self.vfs1, self.vfs2],
            nframes=200,
            analysispath=self.tempdirname1,
            overwrite=True,
            archived=False,
        )
        self.assertTrue(Path.exists(self.tempdirname1 / "movement_video1/coords.darr"))
        self.assertTrue(Path.exists(self.tempdirname1 / "movement_video2/coords.darr"))

    def test_batcharchive(self):
        md.batch_detect_movement(
            [self.vfs1, self.vfs2],
            nframes=200,
            analysispath=self.tempdirname1,
            overwrite=True,
            archived=True,
        )
        self.assertTrue(
            Path.exists(self.tempdirname1 / "movement_video1/coords.darr.tar.xz")
        )
        self.assertTrue(
            Path.exists(self.tempdirname1 / "movement_video2/coords.darr.tar.xz")
        )
