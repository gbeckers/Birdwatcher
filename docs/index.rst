.. Birdwatcher documentation master file, created by
   sphinx-quickstart on Sat Dec 15 07:58:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Birdwatcher
===========

Birdwatcher is a Python computer vision library for the analysis of animal
behavior.

It is developed by Gabriel Beckers, Sita ter Haar and Carien Mol, at
Experimental Psychology, Utrecht University.

It is being used in our lab but not stable enough yet for general use. More
info will be provided when a first release is appropriate.

Code can be found on GitHub: https://github.com/gbeckers/Birdwatcher

It is open source and freely available under the `New BSD License
<https://opensource.org/licenses/BSD-3-Clause>`__ terms.


Installation Birdwatcher package
--------------------------------

Birdwatcher depends on Python 3.6 or higher, and a number of libraries. As
long as there is no official release:

Install dependencies::

    $ pip install darr
    $ pip install opencv-python
    $ pip install opencv-contrib-python

For some functionality it is necessary to install ffmpeg. In Anaconda this can
be done as follows::

    $ conda install -c conda-forge ffmpeg

The package at conda-forge has h264 encoding, which is nice to have.

Then, install Birdwatcher from git repo::

    $ pip install git+https://github.com/gbeckers/birdwatcher



Installation full analysis environment
--------------------------------------

1) Install Anaconda from https://www.anaconda.com/ .

2) Open Anaconda prompt in terminal.

3) Create new environment for Birdwatcher (name is up to you, in example
   here 'mybirdwatcher'). We install Jupter lab and git at the same time::

    $ conda create -n mybirdwatcher python=3.6 jupyterlab git


4) Switch to this new environment:

Linux and MacOS::

$ source activate mybirdwatcher

Windows::

$ conda activate mybirdwatcher

5) Install darr, opencv-python, opencv-contrib-python::

    $ pip install darr
    $ pip install opencv-python
    $ pip install opencv-contrib-python

6) Install ffmpeg::

    $ conda install -c conda-forge ffmpeg

7) Install Birdwatcher from git repo::

    $ pip install git+https://github.com/gbeckers/birdwatcher


Test
----

To run the test suite:

.. code:: python

    >>>import birdwatcher as bw
    >>>bw.test()
    .............
    ----------------------------------------------------------------------
    Ran 13 tests in 2.193s

    OK

    <unittest.runner.TextTestResult run=13 errors=0 failures=0>

Examples
--------

See `jupyter notebook directory on github
<https://github .com/gbeckers/Birdwatcher/tree/master/notebooks>`__ .


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
