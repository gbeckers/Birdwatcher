Birdwatcher
===========

|Repo Status| |Travis Status| |Appveyor Status| |Docs Status|

Birdwatcher is a Python computer vision library for the measurement and
analysis of animal behavior.

The purpose of this library is to provide base functionality for analysing
animal behavior in a Python scientific computing environment. It
is not intended as a final product to analyse specific behaviors of
specific animals, but rather to facilitate working efficiently with
video data in a scientific Python environment and to apply computer vision
algorithms. It can function as a base package for more specific analysis
code or specialized measurement tools.

Despite its name, Birdwatcher is not only for birds. We also successfully
analyze dog behavior, and it could be used on anything that moves. It is
being used in our lab but not stable enough yet for general use. More info
will be provided when a first release is appropriate.

Code can be found on GitHub: https://github.com/gbeckers/Birdwatcher .

An incipient documentation can be found at https://birdwatcher.readthedocs.io .

It is developed by Gabriel Beckers and Carien Mol, at Experimental Psychology,
Utrecht University.

It is open source and freely available under the `New BSD License
<https://opensource.org/licenses/BSD-3-Clause>`__ terms.

Installation Birdwatcher package
--------------------------------

Birdwatcher depends on Python 3.6 or higher, and a number of libraries. As
long as there is no official release. It is best to use the github master
branch. The older (alpha) versions on PyPi are outdated.

Install dependencies::

    $ pip install darr
    $ pip install opencv-python
    $ pip install opencv-contrib-python

It is also necessary to install ffmpeg. If you do not have this already, one
way of getting it is in Anaconda, as follows::

    $ conda install ffmpeg

The package at conda-forge has h264 encoding, which is nice to have.

Then, install the master branch of Birdwatcher from git repo::

    $ pip install git+https://github.com/gbeckers/birdwatcher@master

Or, the development branch::

    $ pip install git+https://github.com/gbeckers/birdwatcher@develop


Installation full analysis environment
--------------------------------------

1) Install Anaconda from https://www.anaconda.com/ .

2) Open Anaconda prompt in terminal.

3) Create new environment for Birdwatcher (name is up to you, in example
   here 'mybirdwatcher'). We install Jupter lab and git at the same time::

    $ conda create -n mybirdwatcher python=3.8 jupyterlab git


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

    $ conda install -c ffmpeg

7) Install Birdwatcher master branch from git repo::

    $ pip install git+https://github.com/gbeckers/birdwatcher@master

Or, the development branch::

    $ pip install git+https://github.com/gbeckers/birdwatcher@develop



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

Contributions
-------------
Sita ter Haar and Dylan Minekus helped exploring the application of movement
detection algorithms.

.. |Repo Status| image:: https://www.repostatus.org/badges/latest/wip.svg
   :alt: Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.
   :target: https://www.repostatus.org/#wip

.. |Travis Status| image:: https://travis-ci.com/gbeckers/Birdwatcher.svg?branch=master
   :target: https://www.travis-ci.com/gbeckers/Birdwatcher?branch=master

.. |Appveyor Status| image:: https://ci.appveyor.com/api/projects/status/github/gbeckers/darr?svg=true
   :target: https://ci.appveyor.com/project/gbeckers/birdwatcher

.. |Docs Status| image:: https://readthedocs.org/projects/birdwatcher/badge/?version=latest
   :target: https://birdwatcher.readthedocs.io/en/latest/

