Loading results movement detection
==================================

.. currentmodule:: birdwatcher

Movement detection fuctions return three results: the ``coords``,
``coordscount`` and ``coordsmean``, which are numpy-like arrays that can
immediately be used for further analyses. However,  they directly map to
disk-based data folders saved next to the video and can also be loaded in a
later session without having to run the detection algoritm again.

::

    myvideo.mp4 (file)
    myvideo (folder)
    ├── coords.darr (folder)
    ├── coordscount.darr (folder)
    ├── coordmean.darr (folder)
    └── movementvideo.mp4 (file)

To read the ``coords`` data in a later session, use the
:class:`CoordinateArrays` class:

.. code:: python

    >>> import birdwatcher as bw
    >>> ca = bw.CoordinateArrays('myvideo/coords.darr/')
    >>> ca
    RaggedArray (497 subarrays with atom shape (2,), r)
    >>> ca.show(framerate=25) # shows the results as a video on screen

The other two disk-based arrays are derived from the ``coords`` data and are
plain Darr arrays:

.. code:: python

    >>> import darr as da
    >>> cm = da.open('movement_zf20s_low/coordsmean.darr/')
    >>> cm[150:160] # get means of frames 150 to 160
    array([[517.34666667, 390.72666667],
           [523.04255319, 392.50483559],
           [538.40122511, 386.70444104],
           [559.11492418, 375.74700718],
           [600.23291367, 366.65827338],
           [650.84895833, 372.53559028],
           [683.68726236, 372.55323194],
           [714.62362972, 384.35809988],
           [742.3156708 , 403.63472379],
           [747.93397231, 403.20979766]])

`Darr <https://github.com/gbeckers/Darr>`__ is a separate library that has been
installed when you installed Birdwatcher. Darr arrays are disk-based but can
be indexed with numpy indexing methods, which return numpy arrays.