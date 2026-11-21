import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from birdwatcher.frames import framecolor, framegray
from birdwatcher.plotting import imshow_frame


class TestImshowFrame(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_imshowcolor(self):
        frame = framecolor(height=480, width=640, color=(1, 2, 3))
        imshow_frame(frame)

    def test_imshowgray(self):
        frame = framegray(height=480, width=640)
        imshow_frame(frame)

    def test_figwithoutax(self):
        frame = framegray(height=48, width=64)
        fig = plt.figure()
        rfig, rax = imshow_frame(frame, fig=fig)
        self.assertIs(rfig, fig)
        self.assertIn(rax, fig.axes)
        self.assertEqual(len(rax.images), 1)

    def test_axwithoutfig(self):
        frame = framegray(height=48, width=64)
        fig, ax = plt.subplots()
        rfig, rax = imshow_frame(frame, ax=ax)
        self.assertIs(rax, ax)
        self.assertIs(rfig, fig)
        self.assertEqual(len(plt.get_fignums()), 1)
        self.assertEqual(len(ax.images), 1)

    def test_figandax(self):
        frame = framegray(height=48, width=64)
        fig, axes = plt.subplots(1, 2)
        rfig, rax = imshow_frame(frame, fig=fig, ax=axes[1])
        self.assertIs(rfig, fig)
        self.assertIs(rax, axes[1])
        self.assertEqual(len(axes[0].images), 0)
        self.assertEqual(len(axes[1].images), 1)

    def test_defaultfigsizeaspect(self):
        frame = framegray(height=480, width=640)
        fig, ax = imshow_frame(frame)
        w, h = fig.get_size_inches()
        self.assertAlmostEqual(w, 14)
        self.assertAlmostEqual(h, 14 * 480 / 640)

    def test_figsize(self):
        frame = framegray(height=480, width=640)
        fig, ax = imshow_frame(frame, figsize=(4, 3))
        self.assertEqual(tuple(fig.get_size_inches()), (4, 3))

    def test_drawrectangle(self):
        frame = framegray(height=48, width=64)
        fig, ax = imshow_frame(frame, draw_rectangle=(10, 30, 5, 45))
        rects = [p for p in ax.patches if isinstance(p, patches.Rectangle)]
        self.assertEqual(len(rects), 1)
        rect = rects[0]
        self.assertEqual(rect.get_xy(), (5, 10))
        self.assertEqual(rect.get_width(), 40)
        self.assertEqual(rect.get_height(), 20)
