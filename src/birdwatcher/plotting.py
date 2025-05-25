"""This module provides convenient functions for plotting video-related data.

These are based on Matplotlib. If you want to display things automatically
in Jupyter, use `%matplotlib inline`

"""

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches


__all__ = ['imshow_frame']


def imshow_frame(frame, fig=None, ax=None, figsize=None, cmap=None,
                 draw_rectangle=None ):
    """Create an matplotlib image plot of the frame.

    Parameters
    ----------
    frame : numpy array image
        Video frame.
    fig : matplotlib Figure object, optional
        Provide if you already have a figure in which the frame should be
        plotted.
    ax : matplotlib Axes object, optional
        Provide if you already have an axes in which the frame should be
        plotted.
    figsize : (float, float), optional
        Width, height in inches. If not provided, defaults to
        rcParams["figure.figsize"] = [6.4, 4.8].
    draw_rectangle : (int, int, int, int), optional
        Draw a rectangle on image. h1, h2, w1, w2. Origin is left top.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The matplotlib Figure and Axes objects.

    """
    if fig is None:
        w = 14
        h = frame.shape[1] * w / frame.shape[0]
        fig, ax = plt.subplots(figsize=(w,h))
    if ax is None:
        fig.add_axes()
    if cmap is None:
        if frame.ndim == 3:
            cmap = None
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        elif frame.ndim ==2:
            cmap = 'gray'
    ax.imshow(frame.astype('uint8'), cmap=cmap)
    if draw_rectangle is not None:
        h1,h2,w1,w2 = draw_rectangle
        x,y, width, height = w1, h1, w2-w1, h2-h1 # we convert to mtpll coords
        rect = patches.Rectangle((x,y), width, height, linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    return fig, ax