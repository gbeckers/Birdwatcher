###
API
###



Video input from file
=====================

.. automodule:: birdwatcher.video

.. autoclass:: birdwatcher.VideoFile
   :members:
   :inherited-members:

.. autoclass:: birdwatcher.VideoFileStream
   :members:
   :inherited-members:

.. autofunction:: birdwatcher.testvideostreamsmall


Frame processing
================

.. currentmodule:: birdwatcher

.. automodule:: birdwatcher.frames

.. autoclass:: birdwatcher.Frames
   :members:
   :inherited-members:

.. autoclass:: birdwatcher.FramesColor
   :members:

.. autoclass:: birdwatcher.FramesGray
   :members:

Movement detection
==================

.. automodule:: birdwatcher.movementdetection.movementdetection

.. autofunction:: birdwatcher.movementdetection.batch_detect_movement

.. autofunction:: birdwatcher.movementdetection.detect_movement

.. autofunction:: birdwatcher.movementdetection.apply_settings

.. autofunction:: birdwatcher.movementdetection.create_movementvideo

Parameter selection
-------------------

.. automodule:: birdwatcher.movementdetection.parameterselection

.. autoclass:: birdwatcher.movementdetection.ParameterSelection
   :members:
   :inherited-members:

.. autofunction:: birdwatcher.movementdetection.apply_all_parameters

.. autofunction:: birdwatcher.movementdetection.load_parameterselection


Coordinate Arrays
=================

.. automodule:: birdwatcher.coordinatearrays

.. autoclass:: birdwatcher.CoordinateArrays
   :members:
   :inherited-members:

.. autofunction:: open_archivedcoordinatedata

.. autofunction:: create_coordarray



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


Plotting
========

.. automodule:: birdwatcher.plotting

.. autofunction:: birdwatcher.plotting.imshow_frame


Utils
=====

.. automodule:: birdwatcher.utils

.. autofunction:: birdwatcher.utils.product_dict

.. autofunction:: birdwatcher.utils.tempdir

.. autofunction:: birdwatcher.utils.derive_filepath

.. autofunction:: birdwatcher.utils.print_dirstructure

.. autofunction:: birdwatcher.utils.walk_paths