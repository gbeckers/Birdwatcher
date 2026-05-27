Concatenating transformations
=============================

.. currentmodule:: birdwatcher

When applying multiple transformations to a video, it is best to concatenate
them on a :class:`Frames` object:

.. code:: python

    >>> import birdwatcher as bw
    >>> vfs = bw.VideoFileStream('myvideo.mp4')
    >>> (vfs.iter_frames()
    ...    .resize((640,360))
    ...    .togray()
    ...    .blur(ksize=(5,5))
    ...    .draw_framenumbers()
    ...    .tovideo('myvideo_resized', framerate=vfs.avgframerate))

The 1) decodes the video (``iter_frames``), 2) resizes the video (``resize``)
, 3) converts color to gray values (``togray``), 4) blurs frame images
(``blur``), 5) draws frame numbers on frames (``draw_framenumbers``), and 6)
encodes and saves the video (``tovideo``).

Concatenation avoids repeated decoding and encoding a video, which is
computationally intensive. Also, multiple decoding-transformation-re-encoding
rounds would lead to more loss of information as most video codecs apply lossy
compression.