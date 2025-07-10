############
Installation
############

.. contents:: :local:

Birdwatcher officially supports Python 3.10 or higher, but older
Python 3 versions may also work.

**User installation**

#. We recommend using Anaconda for installation. Install Anaconda from https://www.anaconda.com/ .

#. Open Anaconda prompt in terminal.

#. Create new environment for Birdwatcher (name is up to you, in example here 'mybirdwatcher'). We install Jupyter lab and ffmpeg at the same time:

    .. code-block:: bash

      conda create -n mybirdwatcher python=3.13 jupyterlab git

#. Switch to this new environment:

   .. code-block:: bash

      conda activate mybirdwatcher

#. Install ffmpeg from conda-forge. It needs to be version 7.1.1

    ..code-block:: bash

      conda install -c conda-forge ffmpeg=7.1.1


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
