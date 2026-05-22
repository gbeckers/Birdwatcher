Checking out the properties of a video file
===========================================

.. currentmodule:: birdwatcher.video

To check out the properties of a video file it is good to distinguish between a video *file*, represented by a :class:`VideoFile` object and a video file *stream*, represented by a :class:`VideoFileStream` object.

.. code:: python

    >>> import birdwatcher as bw
    >>> vf = bw.VideoFile('myvideo.mp4')

A video file can contain multiple data streams. Often one video stream and one audio stream, but there could also be multiple video and audio streams. If we are interested in video data in the video, we can check how many streams there are:

.. code:: python

    >>> vf.nvideostreams
    1

There is only one in this case. We can obtain it using the :meth:`VideoFile.get_videostream` method.

.. code:: python

    >>> vfs = vf.get_videofilestream(0)

The resulting :class:`VideoFileStream` object is a starting point for many computer vision analyses in Birdwatcher.

You may also be interested in audio data in the videofile.

.. code:: python

    >>> vf.naudiostreams
    1

Birdwatcher doesn't help you analyse it, but it can extract the audio info and save it as an audio file for analysis in other programs. See the :doc:`extractaudio` recipe.

You can check all known properties of a videofile through the :attr:`streamsinfo` and :attr:`formatinfo` attributes. E.g.:

.. code:: python

    >>> vf.formatinfo
    {'filename': 'myvideo.mp4',
     'nb_streams': 1,
     'nb_programs': 0,
     'nb_stream_groups': 0,
     'format_name': 'mov,mp4,m4a,3gp,3g2,mj2',
     'format_long_name': 'QuickTime / MOV',
     'start_time': '0.000000',
     'duration': '19.880000',
     'size': '5541582',
     'bit_rate': '2230012',
     'probe_score': 100,
     'tags': {'major_brand': 'isom',
              'minor_version': '512',
              'compatible_brands': 'isomiso2avc1mp41',
              'encoder': 'Lavf58.12.100'}}



