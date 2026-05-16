Detecting movement in a video
=============================

There is a high-level function, `detect_movement`, that makes this simple:

.. code:: python

    >>> import birdwatcher as bw
    >>> import birdwatcher.movementdetection as md
    >>> vfs = bw.VideoFileStream('myvideo.mp4')
    >>> coords, coordscount, coordsmean = md.detect_movement(vfs,
    ...    ignore_firstnframes=50, resultvideo=True)

This will create a subdirectory `myvideo` in which the the results will be
saved. This includes a video showing the positive pixels superimposed on the
original video. The actual results, `coords`, `coordscount` and `coordsmean`
arrays, are returned from the `detect_movement` function for further
evaluation, but the data itself is disk-based and stored in the subdirectory.
This way, the results can be loaded in a later session, without running the
`detect_movement` function again.


