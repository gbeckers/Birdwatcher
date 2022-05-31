###########
Development
###########

.. contents:: :local:

Contributing
============

If you would like to contribute to Birdwatcher, please follow the Style Guide conventions for consistency of code and documentation.

Style guide
===========

Here, you can find a description of the code and documentation conventions used throughout Birdwatcher. 

In general, we follow the standard Python style conventions for our code:

- `PEP 8 - Style Guide for Python Code <https://peps.python.org/pep-0008/>`__
- `PEP 257 - Docstring conventions <https://peps.python.org/pep-0257/>`__

The documentation of Birdwatcher (see https://birdwatcher.readthedocs.io) is written in `re-structured text (reST) <https://docutils.sourceforge.io/rst.html>`__ syntax and is rendered using `Sphinx <https://www.sphinx-doc.org/en/master/>`__.

For more precise docstring conventions, we follow the `Docstring Standard of Numpy <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`__. Also, see below for some highlighted information and examples.

Docstring Parameters
--------------------

In the parameter description of the docstring, specify behind each parameter its ``type``. Add whether a keyword parameter is ``optional``. Optional parameters always have default values. 

The **default value** can be described in several ways:

1) For more simple, straightforward default values, add ``default=value`` instead of ``optional``, behind the type description. See ``color`` and ``reportprogress`` in the example_.

2) Describe the default value in the detailed description of parameter if more information is necessary. Make sure to also add ``optional`` behind the type description. See ``ffmpegpath`` in the example_.

3) Use ``optional`` when the default value would not be used as value. See ``startat`` and ``nframes`` in the example_.

.. _example:

**An example:**

.. code:: python

    def iter_frames(self, startat=None, nframes=None, color=True,
                    ffmpegpath='ffmpeg', reportprogress=False):
        """Iterate over frames in video.

        Parameters
        ----------
        startat : str, optional
            If specified, start at this time point in the video file. You
            can use two different time unit formats: sexagesimal
            (HOURS:MM:SS.MILLISECONDS, as in 01:23:45.678), or in seconds.
        nframes  : int, optional
            Read a specified number of frames.
        color : bool, default=True
            Read as a color frame (3 dimensional) or as a gray frame (2
            dimensional).
        ffmpegpath : str or pathlib.Path, optional
            Path to ffmpeg executable. Default is `ffmpeg`, which means it
            should be in the system path.
        reportprogress : bool, default=False    
        
        Yields
        ------
        Frames
            Iterator that generates numpy array frames (height x width x color 
            channel).
            
        """