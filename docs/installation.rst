############
Installation
############

.. contents:: :local:

Birdwatcher officially supports Python 3.12 or higher, but older
Python 3 versions may also work.

**User installation**

#. We recommend using Anaconda for installation. Install Anaconda from https://www.anaconda.com/ .

#. Open Anaconda prompt in terminal.

#. Create new environment for Birdwatcher (name is up to you, in example here 'mybirdwatcher'). We install Jupyter lab and ffmpeg at the same time:

   .. code-block:: bash

     conda create -n mybirdwatcher python=3.14 git=2.51 ffmpeg=8.0.1 numpy=2.4.2 matplotlib=3.10.8 pandas=3.0.0 seaborn=0.13.2

#. Switch to this new environment:

   .. code-block:: bash

      conda activate mybirdwatcher

#. Install jupyter from conda-forge (currently conda one has a problem):

   .. code-block:: bash

      conda install -c conda-forge jupyter


#. Install Birdwatcher:

   Stable latest official release from PyPi:

   .. code-block:: bash

      pip install Birdwatcher

   If instead you want the latest version of the git master branch from, use:

   .. code-block:: bash

      pip install git+https://github.com/gbeckers/birdwatcher@master


**Dependencies**

The following dependencies are automatically taken care of when you
install Birdwatcher using the pip method above:

- numpy
- pandas
- matplotlib
- seaborn
- darr
- opencv-python
- opencv-contrib-python

It further depends on:

- ffmpeg (including ffprobe)

If you do not use the conda way above to install ffmpeg, you need to
install it yourself (https://www.ffmpeg.org/).
