#############################
Birdwatcher API Documentation
#############################

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
frames and many of its methods return another Frames object. Since
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
- crop
- draw_circles
- draw_framenumbers
- draw_rectangles
- draw_text
- find_contours
- find_nonzero
- get_info
- morphologyex
- peek_frame
- resize
- resizebyfactor
- show
- threshold
- tocolor
- togray
- tovideo

.. automodule:: birdwatcher.frames

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


Coordinate Arrays
=================

.. automodule:: birdwatcher.coordinatearrays

.. autoclass:: birdwatcher.CoordinateArrays
   :members:
   :inherited-members:

.. autofunction:: open_archivedcoordinatedata

.. autofunction:: create_coordarray


Movement detection
==================

Movement detection contains top-level functionality. The classes, methods and functions provided in these submodules, are written to help the user get the most out of Birdwatcher. Also see the notebooks for examples how to use it and  how to find the optimal parameter settings for movement detection.

.. automodule:: birdwatcher.movementdetection.movementdetection

.. autofunction:: birdwatcher.movementdetection.detect_movement

.. autofunction:: birdwatcher.movementdetection.detect_movementmog2

.. autofunction:: birdwatcher.movementdetection.detect_movementknn

.. autofunction:: birdwatcher.movementdetection.detect_movementlsbp

.. autofunction:: birdwatcher.movementdetection.create_movementvideo

Parameter selection
-------------------

.. automodule:: birdwatcher.movementdetection.parameters

.. autoclass:: birdwatcher.movementdetection.ParameterSelection
   :members:
   :inherited-members:

.. autofunction:: birdwatcher.movementdetection.apply_all_parameters

.. autofunction:: birdwatcher.movementdetection.load_parameterselection


Plotting
========

.. automodule:: birdwatcher.plotting

.. autofunction:: birdwatcher.plotting.imshow_frame


Utils
=====

.. automodule:: birdwatcher.utils

.. autofunction:: birdwatcher.utils.walk_paths

