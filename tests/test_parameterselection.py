import copy
import io
import json
import unittest
import tempfile
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import birdwatcher as bw
import birdwatcher.movementdetection as md


settings = {
    "bgs_params": {
        "History": [4, 12],
        "ComplexityReductionThreshold": [0.05],
        "BackgroundRatio": [0.1],
        "NMixtures": [7],
        "VarInit": [15],
        "VarMin": [4],
        "VarMax": [75],
        "VarThreshold": [10],
        "VarThresholdGen": [9],
        "DetectShadows": [False],
        "ShadowThreshold": [0.5],
        "ShadowValue": [0],
    },
    "processing": {
        "color": [True],
        "resizebyfactor": [1],
        "blur": [
            0,
        ],
        "morphologyex": [True],
    },
}


def _fresh_copy(params):
    """Returns a new ParameterSelection with a copy of the data, so that
    tests do not influence each other without recomputing everything."""
    return md.ParameterSelection(
        params.df.copy(),
        params.vfs.filepath,
        params.bgs_type,
        params.startat,
        params.nframes,
        params.roi,
        params.nroi,
        params.path,
    )


class TestParameterSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._params = md.apply_all_parameters(
            bw.testvideostreamsmall(), settings, nframes=200, reportprogress=False
        )

    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        self.params = _fresh_copy(self._params)

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)

    def test_attributes(self):
        self.assertIsInstance(self.params, md.ParameterSelection)
        self.assertIsInstance(self.params.df, pd.DataFrame)
        self.assertIsInstance(self.params.vfs, bw.VideoFileStream)
        self.assertEqual(self.params.bgs_type, str(bw.BackgroundSubtractorMOG2))
        self.assertEqual(self.params.startat, None)
        self.assertEqual(self.params.nframes, 200)
        self.assertEqual(self.params.path, None)

    def test_getvideofragment(self):
        frames = self.params.get_videofragment()
        self.assertIsInstance(frames, bw.Frames)

    def test_getparams(self):
        all = self.params.get_parameters("all")
        settings_flat = {**settings["bgs_params"], **settings["processing"]}
        self.assertEqual(all, settings_flat)
        multi_only = self.params.get_parameters("multi_only")
        self.assertEqual(multi_only, {"History": [4, 12]})

    def test_plotparams(self):
        g = self.params.plot_parameters(
            rows="History", cols=None, default_values={"History": 4}
        )
        self.assertIsInstance(g, sns.axisgrid.FacetGrid)
        self.assertEqual(g.axes.shape, (2, 1))
        plt.close(g.figure)

    def test_drawcircles(self):
        frames, colorspecs = self.params.draw_multiple_circles({"History": [4, 12]})
        self.assertIsInstance(frames, bw.Frames)
        self.assertEqual(colorspecs.shape, (1, 2))

    def test_saveparams(self):
        self.assertIsNone(self.params.path)
        self.params.save_parameters(self.tempdirname1)
        self.assertIsNotNone(self.params.path)

    def test_saveanotherparams(self):
        self.params.save_parameters(self.tempdirname1)
        self.params.save_parameters(self.tempdirname1)
        self.assertEqual(self.params.path[-1], "2")

    def test_dataframe_shape(self):
        # 200 frames x 2 History settings x 2 coords (x, y)
        self.assertEqual(len(self.params.df), 200 * 2 * 2)
        self.assertEqual(sorted(self.params.df.coords.unique()), ["x", "y"])
        self.assertEqual(
            sorted(self.params.df.framenumber.unique()), list(range(200))
        )
        self.assertIn("pixel", self.params.df.columns)

    def test_getinfo(self):
        info = self.params.get_info()
        self.assertEqual(
            set(info), {"vfs", "bgs_type", "startat", "nframes", "roi", "nroi"}
        )
        self.assertEqual(info["vfs"], str(self.params.vfs.filepath))
        self.assertEqual(info["nframes"], 200)
        # info should be serializable, because it is stored in the csv file
        json.dumps(info)

    def test_getparams_invalidselection(self):
        with self.assertRaises(Exception):
            self.params.get_parameters("nonexisting")

    def test_plotparams_missingdefaults(self):
        with self.assertRaises(Exception):
            self.params.plot_parameters(
                rows="History", cols=None, default_values={}
            )

    def test_drawcircles_missingparameter(self):
        with self.assertRaises(Exception):
            self.params.draw_multiple_circles({"blur": [0]})

    def test_drawcircles_toomanycombinations(self):
        n = len(md.ParameterSelection.colors) + 1
        with self.assertRaises(Exception):
            self.params.draw_multiple_circles({"History": list(range(n))})

    def test_drawcircles_singlesetting(self):
        frames, colorspecs = self.params.draw_multiple_circles({"History": [12]})
        self.assertEqual(colorspecs.shape, (1, 1))
        self.assertEqual(list(colorspecs.columns), ["orange"])
        frame = next(iter(frames))
        self.assertEqual(frame.ndim, 3)

    def test_saveparams_files(self):
        self.params.save_parameters(self.tempdirname1)
        path = Path(self.params.path)
        self.assertEqual(path.name, f"params_{self.params.vfs.filepath.stem}")
        self.assertTrue((path / "parameterselection.csv").exists())
        readme = (path / "readme.txt").read_text()
        self.assertIn("History: [4, 12]", readme)
        self.assertEqual(
            len(readme.splitlines()), len(self.params.get_parameters("all"))
        )

    def test_saveparams_foldername(self):
        self.params.save_parameters(self.tempdirname1, foldername="myparams")
        self.assertEqual(Path(self.params.path), self.tempdirname1 / "myparams")

    def test_saveparams_overwrite(self):
        self.params.save_parameters(self.tempdirname1, foldername="p")
        self.params.save_parameters(self.tempdirname1, foldername="p", overwrite=True)
        self.assertEqual(Path(self.params.path), self.tempdirname1 / "p")
        self.assertFalse((self.tempdirname1 / "p_2").exists())

    def test_loadparams(self):
        self.params.save_parameters(self.tempdirname1)
        loaded = md.load_parameterselection(self.params.path)
        self.assertIsInstance(loaded, md.ParameterSelection)
        pd.testing.assert_frame_equal(loaded.df, self.params.df)
        self.assertEqual(loaded.path, self.params.path)
        self.assertEqual(loaded.get_info(), self.params.get_info())
        self.assertEqual(
            loaded.get_parameters("all"), self.params.get_parameters("all")
        )

    def test_batchplot_singleparameter(self):
        # with only one multi-value parameter there are no pairs to plot, but
        # the folder and default values file should still be created
        self.params.path = str(self.tempdirname1)
        with redirect_stdout(io.StringIO()):
            self.params.batch_plot_parameters({"History": 4})
        figpath = self.tempdirname1 / "figures"
        self.assertEqual(
            (figpath / "default_values.txt").read_text(), "History: 4\n"
        )
        self.assertEqual(list(figpath.glob("*.png")), [])


class TestParameterSelectionTwoParams(unittest.TestCase):
    """Parameter selection with two parameters tested with multiple values."""

    @classmethod
    def setUpClass(cls):
        cls.settings = copy.deepcopy(settings)
        cls.settings["processing"]["blur"] = [0, 5]
        cls._params = md.apply_all_parameters(
            bw.testvideostreamsmall(),
            cls.settings,
            nframes=20,
            roi=(10, 100, 20, 200),
            reportprogress=False,
        )

    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        self.params = _fresh_copy(self._params)

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)

    def test_getparams(self):
        multi_only = self.params.get_parameters("multi_only")
        self.assertEqual(multi_only, {"History": [4, 12], "blur": [0, 5]})

    def test_roi(self):
        self.assertEqual(self.params.roi, (10, 100, 20, 200))
        self.assertEqual(self.params.get_info()["roi"], (10, 100, 20, 200))

    def test_plotparams(self):
        g = self.params.plot_parameters(
            rows="History", cols="blur", default_values={"History": 4, "blur": 0}
        )
        self.assertEqual(g.axes.shape, (2, 2))
        plt.close(g.figure)

    def test_plotparams_onlyrows(self):
        g = self.params.plot_parameters(
            rows="History", cols=None, default_values={"History": 4, "blur": 5}
        )
        self.assertEqual(g.axes.shape, (2, 1))
        self.assertIn("'blur': 5", g.figure._suptitle.get_text())
        plt.close(g.figure)

    def test_drawcircles(self):
        frames, colorspecs = self.params.draw_multiple_circles(
            {"History": [4, 12], "blur": [0, 5]}
        )
        self.assertIsInstance(frames, bw.Frames)
        self.assertEqual(colorspecs.shape, (2, 4))
        self.assertEqual(
            list(colorspecs.columns), [c[0] for c in md.ParameterSelection.colors[:4]]
        )

    def test_batchplot(self):
        self.params.path = str(self.tempdirname1)
        with redirect_stdout(io.StringIO()):
            self.params.batch_plot_parameters({"History": 4, "blur": 0})
            self.params.batch_plot_parameters({"History": 12, "blur": 5})
        stem = self.params.vfs.filepath.stem
        for foldername, defaults in [
            ("figures", "History: 4\nblur: 0\n"),
            ("figures_2", "History: 12\nblur: 5\n"),
        ]:
            figpath = self.tempdirname1 / foldername
            self.assertEqual((figpath / "default_values.txt").read_text(), defaults)
            self.assertEqual(
                sorted(p.name for p in figpath.glob("*.png")),
                [f"{stem}_History_blur.png", f"{stem}_blur_History.png"],
            )

    def test_batchplot_overwrite(self):
        self.params.path = str(self.tempdirname1)
        with redirect_stdout(io.StringIO()):
            self.params.batch_plot_parameters({"History": 4, "blur": 0})
            self.params.batch_plot_parameters(
                {"History": 12, "blur": 0}, overwrite=True
            )
        self.assertFalse((self.tempdirname1 / "figures_2").exists())
        self.assertEqual(
            (self.tempdirname1 / "figures" / "default_values.txt").read_text(),
            "History: 12\nblur: 0\n",
        )


class TestApplyAllParameters(unittest.TestCase):
    def test_invalid_usestats(self):
        with self.assertRaises(Exception):
            md.apply_all_parameters(
                bw.testvideostreamsmall(),
                settings,
                nframes=5,
                use_stats="median",
                reportprogress=False,
            )

    def test_reportprogress(self):
        f = io.StringIO()
        with redirect_stdout(f):
            md.apply_all_parameters(
                bw.testvideostreamsmall(), settings, nframes=5, reportprogress=1
            )
        lines = f.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("1 combinations of settings applied"))
        self.assertTrue(lines[1].startswith("2 combinations of settings applied"))

    def test_reportprogress_off(self):
        f = io.StringIO()
        with redirect_stdout(f):
            md.apply_all_parameters(
                bw.testvideostreamsmall(), settings, nframes=5, reportprogress=False
            )
        self.assertEqual(f.getvalue(), "")

    def test_startat(self):
        params = md.apply_all_parameters(
            bw.testvideostreamsmall(),
            settings,
            startat="00:00:01",
            nframes=5,
            nroi=(0, 10, 0, 10),
            reportprogress=False,
        )
        self.assertEqual(params.startat, "00:00:01")
        self.assertEqual(params.nroi, (0, 10, 0, 10))
        self.assertEqual(len(params.df), 5 * 2 * 2)


class TestParamSelectionCount(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._params = md.apply_all_parameters(
            bw.testvideostreamsmall(),
            settings,
            nframes=200,
            use_stats="count",
            reportprogress=False,
        )

    def setUp(self):
        self.tempdirname1 = Path(tempfile.mkdtemp())
        self.params = _fresh_copy(self._params)

    def tearDown(self):
        shutil.rmtree(self.tempdirname1)

    def test_count(self):
        coordslabels = self.params.df.coords.unique()
        self.assertEqual(coordslabels.shape, (1,))
        self.assertEqual(coordslabels[0], "count")

    def test_dataframe_shape(self):
        # 200 frames x 2 History settings x 1 count
        self.assertEqual(len(self.params.df), 200 * 2)
        self.assertTrue((self.params.df.pixel >= 0).all())

    def test_drawcircles_raises(self):
        with self.assertRaises(Exception):
            self.params.draw_multiple_circles({"History": [4, 12]})

    def test_plotparams(self):
        g = self.params.plot_parameters(
            rows="History", cols=None, default_values={"History": 4}
        )
        self.assertEqual(g.axes.shape, (2, 1))
        plt.close(g.figure)

    def test_loadparams(self):
        self.params.save_parameters(self.tempdirname1)
        loaded = md.load_parameterselection(self.params.path)
        pd.testing.assert_frame_equal(loaded.df, self.params.df)
