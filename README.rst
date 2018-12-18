Birdwatcher
===========

|Repo Status| |Travis Status| |Appveyor Status| |Docs Status|

Birdwatcher is a Python computer vision library for the analysis of animal
behavior.

It is developed by Gabriel Beckers, Sita ter Haar and Carien Mol, at
Experimental Psychology, Utrecht University.

It is being used in our lab but not stable enough yet for general use. More
info will be provided when a first release is appropriate.

Code can be found on GitHub: https://github.com/gbeckers/Birdwatcher .

An incipient documentation can be found at https://birdwatcher.readthedocs.io .

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

It is also necessary to install ffmpeg. If you do not have this already, one
way of getting it is in Anaconda, as follows::

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


Documentation
-------------

https://birdwatcher.readthedocs.io


Examples
--------

See notebook directory.

.. |Repo Status| image:: https://www.repostatus.org/badges/latest/wip.svg
   :alt: Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.
   :target: https://www.repostatus.org/#wip

.. |Travis Status| image:: https://travis-ci.org/gbeckers/Birdwatcher.svg?branch=master
   :target: https://travis-ci.org/gbeckers/Birdwatcher?branch=master

.. |Appveyor Status| image:: https://ci.appveyor.com/api/projects/status/github/gbeckers/darr?svg=true
   :target: https://ci.appveyor.com/project/gbeckers/birdwatcher

.. |Docs Status| image:: https://readthedocs.org/projects/birdwatcher/badge/?version=latest
   :target: https://birdwatcher.readthedocs.io/en/latest/

