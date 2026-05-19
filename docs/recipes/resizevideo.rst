Resize video
============

Use the :meth:`resize` method, which takes a tuple with the desired size as
an argument. Let's resize to (640,360).
.. code:: python

    >>> import birdwatcher as bw
    >>> vfs = bw.VideoFileStream('myvideo.mp4')
    >>> (vfs.iter_frames()
    ...    .resize((640,360))
    ...    .tovideo('myvideo_resized', framerate=vfs.avgframerate))

Alternatively you can resize by specifying a factor, e.g. 0.5, using the
:meth:`resizebyfactor` method.

Much of the functionality in terms of video transformations is accessible
through :class:`Frames` objects, which yield video frames. A Frames
object can be obtained from a :class:`VideoFileStream` object using the
:meth:`iter_frames` method. Frames objects have many interesting methods,
among which :meth:`resize`, which yields another Frames object, with resized
frames. Frames objects can be used for analyses, but they can also be
written as a video file using their :meth:`tovideo` method.

More explanation on Frames is provided in our tutorial notebook
`1_videoframes <https://github.com/gbeckers/Birdwatcher/blob/master/notebooks
/1_videoframes.ipynb>`__