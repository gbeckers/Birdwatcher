Add frame numbers
=================

Adding frame numbers to video frames can be handy for precise reference. Use
the :meth:`draw_framenumbers` method on a :class:`Frames` object:

.. code:: python

    >>> import birdwatcher as bw
    >>> vfs = bw.VideoFileStream('myvideo.mp4')
    >>> (vfs.iter_frames()
    ...    .draw_framenumbers()
    ...    .tovideo('myvideo_resized', framerate=vfs.avgframerate))

See :doc:`resizevideo` for some explanation on the :meth:`iter_frames` method.
For even more explanation see our tutorial notebook `1_videoframes
<https://github.com/gbeckers/Birdwatcher/blob/master/notebooks/1_videoframes
.ipynb>`__