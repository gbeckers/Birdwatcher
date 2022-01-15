############
Installation
############

.. contents:: :local:

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

    $ conda install ffmpeg

The package at conda-forge has h264 encoding, which is nice to have.

Then, install Birdwatcher from git repo::

    $ pip install git+https://github.com/gbeckers/birdwatcher



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

    $ conda install ffmpeg

7) Install Birdwatcher from git repo::

    $ pip install git+https://github.com/gbeckers/birdwatcher
