Detecting movement
==================

In many lab situations (static camera and static scene, except for animal),
'movement' can be captured based on detecting pixels in video frames that
changed statistcally significantly with respect to preceding frames.
The high-level function :func:`detect_movement` does this, based on a
machine learning technique called background substraction.

.. code:: python

    >>> import birdwatcher as bw
    >>> import birdwatcher.movementdetection as md
    >>> vfs = bw.VideoFileStream('myvideo.mp4')
    >>> (coords,
    ...  coordscount,
    ...  coordsmean) = md.detect_movement(vfs, resultvideo=True)

This will create a subdirectory `myvideo` in which the results are
saved.

::

    myvideo.mp4 (file)
    myvideo (folder)
    ├── coords.darr (folder)
    ├── coordscount.darr (folder)
    ├── coordmean.darr (folder)
    └── movementvideo.mp4 (file)

The returned results, the ``coords``, ``coordscount`` and ``coordsmean``
are numpy-like arrays that map to the disk-based data folders and can
immediately be used for further analyses, but can also be loaded in a later
session. The included movement video can be used for quick inspection of
results, showing the positive pixels superimposed on the original video. If you
are not happy with the results, you can tweak the detection parameters, if
needed based on an automated procedure to find the settings you need (see
`tutorial notebook <https://github.com/gbeckers/Birdwatcher/blob/master/notebooks
/4_parameterselection.ipynb>`__).

The ``coords`` array contains the coordinates (x,y) of the pixels that are
above the significance threshold for considering them 'changed´. We can for
example see which pixels are detected in video frame number 157:

.. code-block:: python

    >>> coords[157]
    array([[718, 358],
           [719, 358],
           [716, 359],
           ...,
           [705, 410],
           [707, 410],
           [708, 410]], shape=(821, 2), dtype=uint16)



The result is a 2D array with on the first axis pixels (821 have been
detected) and on the second axis spatial dimensions (x and y coordinates). We
can also get these results in a frame where detected pixels have the value 1
and the rest has the value 0. This way we can plot them as an image:

.. code-block:: python

    >>> from birdwatcher.plotting import imshow_frame
    >>> imshow_frame(coords.get_frame(157))

.. figure:: images/flyingbirdcoords.png
  :align: center
  :width: 720
  :class: custom-image

There is a blob of positive pixels roughly around x: 700 to 800 and y: 350 to
450. Let's check that frame in the original video:

.. code-block:: python

    >>> from birdwatcher.plotting import imshow_frame
    >>> imshow_frame(coords.get_frame(157))

.. figure:: images/flyingbirdframe.png
  :align: center
  :width: 720
  :class: custom-image

A bird jumps from one perch to the other. We can quantify the amount of
movement in the frame by counting the number of detected pixels:

.. code-block:: python

    >>> c = coords[157]
    >>> len(c)
    821

The pixel count for every frame is provided by the ``coordscount`` variable
in the :func:`detect_movement` function that we used above. We can plot the
count for all frames:

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(coordscount)
    >>> plt.xlabel('frame number'); plt.ylabel('number of positive pixels')


.. figure:: images/birdcoordscount.png
  :align: center
  :width: 720
  :class: custom-image

Frame number is on the x-axis, count on the y-axis. Similarly, we can look at
the means of the x- and y-coordinates based on the ``coordsmean`` variable
returned by the :func:`detect_movement` function :

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(coordsmean)
    >>> plt.xlabel('frame number'); plt.ylabel('mean'); plt.legend(['x','y' ])

.. figure:: images/birdcoordsmean.png
  :align: center
  :width: 720
  :class: custom-image

Frame number is on the x-axis, space on the y-axis. Blue is the x-coordinate,
orange the y-coordinate. Without having watched the video, we conclude that the
bird sat on perch #3, jumped to perch #4 (on equal height), jumped back, and
then jumped to the higher perch #1. If we know the framerate of the video, we
can calculate exactly when the bird was doing this.