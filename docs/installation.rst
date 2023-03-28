############
Installation
############

.. contents:: :local:

Installation Birdwatcher package
--------------------------------

Birdwatcher officially supports Python 3.9 or higher, but older
Python 3 versions may also work.

**User installation**

1) We recommend using Anaconda for installation. Install Anaconda from https://www.anaconda.com/ .

2) Open Anaconda prompt in terminal.

3) Create new environment for Birdwatcher (name is up to you, in example
   here 'mybirdwatcher'). We install Jupter lab and ffmpeg at the same time::

    $ conda create -n mybirdwatcher python=3.9 jupyterlab ffmpeg

4) Switch to this new environment:

   Linux and MacOS::

    $ source activate mybirdwatcher

   Windows::

    $ conda activate mybirdwatcher

5) Install Birdwatcher master branch from git repo::

    $ pip install git+https://github.com/gbeckers/birdwatcher@master


**Dependencies**

The following dependencies are automatically taken care of when you
install Birdwatcher from GitHub using the pip method above:

- numpy
- matplotlib
- darr
- opencv-python
- opencv-contrib-python

It further depends on:

- ffmpeg (including ffprobe)

If you do not use the conda way above to install it, you need to
install it yourself (https://www.ffmpeg.org/).
