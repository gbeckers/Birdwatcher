#############################
Birdwatcher API Documentation
#############################

.. contents:: :local:
   :depth: 2

Video input from file
=====================

.. automodule:: birdwatcher.video

.. autoclass:: birdwatcher.VideoFileStream
   :members:
   :inherited-members:

.. autofunction:: birdwatcher.testvideosmall

Frame processing
================

`Frames` is a central class in Birdwatcher. It is an iterable that yields
frames and many of its methods return another Frames object. Processing
starts when iteration over it, or when calling its `tovideo` method. Since
there are quite a few methods, we'll list property and method names first,
and then provide detailed info below that.

Frames properties
-----------------
- dtype
- frameheight
- framewidth
- nchannels

Frames methods
--------------
- absdiff_frame
- add_weighted
- apply_backgroundsegmenter
- blur
- calc_meanframe
- draw_circles
- draw_framenumbers
- draw_rectangles
- draw_text
- find_contours
- find_nonzero
- get_info
- morphologyex
- resizes
- resizebyfactor
- threshold
- tocolor
- togray
- tovideo

.. automodule:: birdwatcher.frameprocessing

.. autoclass:: birdwatcher.Frames
   :members:
   :inherited-members:

.. autoclass:: birdwatcher.FramesColor
   :members:

.. autoclass:: birdwatcher.FramesGray
   :members:



Background subtraction
======================

.. automodule:: birdwatcher.backgroundsubtraction

MOG2
----

.. autoclass:: birdwatcher.BackgroundSubtractorMOG2
   :members:
   :inherited-members:

KNN
---

.. autoclass:: birdwatcher.BackgroundSubtractorKNN
   :members:
   :inherited-members:

LSBP
----

.. autoclass:: birdwatcher.BackgroundSubtractorLSBP
   :members:
   :inherited-members:

.. autofunction:: open_archivedcoordinatedata

.. autofunction:: create_coordarray

Movement detection
==================

.. automodule:: birdwatcher.movementdetection

.. autofunction:: birdwatcher.detect_movement

.. autofunction:: detect_movementmog2

.. autofunction:: detect_movementknn

.. autofunction:: detect_movementlsbp

.. autofunction:: create_movementvideo


Coordinate Arrays
=================

.. automodule:: birdwatcher.coordinatearrays

.. autoclass:: birdwatcher.CoordinateArrays
   :members:
   :inherited-members:

.. autofunction:: open_archivedcoordinatedata

.. autofunction:: create_coordarray

.. autofunction::  open_archivedcoordinatedata


Plotting
========

.. automodule:: birdwatcher.plotting

.. autofunction:: birdwatcher.plotting.imshow_frame

Utils
========

.. automodule:: birdwatcher.utils

.. autofunction:: birdwatcher.utils.walk_paths

