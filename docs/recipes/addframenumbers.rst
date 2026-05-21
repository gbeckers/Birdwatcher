Add frame numbers
=================

.. currentmodule:: birdwatcher

Adding frame numbers to video frames can be useful for precise reference. Use
the :meth:`Frames.draw_framenumbers` method on a :class:`Frames` object:

.. code:: python

    >>> import birdwatcher as bw
    >>> vfs = bw.VideoFileStream('myvideo.mp4')
    >>> (vfs.iter_frames()
    ...    .draw_framenumbers()
    ...    .tovideo('myvideo_resized', framerate=vfs.avgframerate))

See :doc:`resizevideo` for some explanation on the
:meth:`VideoFileStream.iter_frames` method. For even more explanation see our
tutorial notebook `1_videoframes <https://github.com/gbeckers/Birdwatcher/
blob/master/notebooks/1_videoframes.ipynb>`__