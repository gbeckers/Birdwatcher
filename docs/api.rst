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


Movement detection
==================

.. automodule:: birdwatcher.movementdetection

.. autoclass:: birdwatcher.MovementDetection
   :members:
   :inherited-members:


.. autofunction:: birdwatcher.detect_movement