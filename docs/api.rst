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


Plotting
========

.. automodule:: birdwatcher.plotting

.. autofunction:: birdwatcher.plotting.imshow_frame

Utils
========

.. automodule:: birdwatcher.utils

.. autofunction:: birdwatcher.utils.walk_paths

