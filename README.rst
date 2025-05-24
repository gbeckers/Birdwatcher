Birdwatcher
===========

|Github CI Status| |PyPi version| |Docs Status| |Repo Status|
|Codecov status|

.. image:: https://github.com/gbeckers/Birdwatcher/blob/master/docs/images/banner.gif
  :align: center
  :width: 720

Birdwatcher is a Python computer vision library for analyzing animal behavior in a Python scientific computing environment.

Birdwatcher should help you getting up and running quickly when building analysis code or tools for specific measurements. It provides functionality that is common in video analysis, such as reading and writing videos into and from numpy arrays, applying processing algorithms such as background subtraction, morphological transformation, resizing, drawing on frames etc. Much of the underlying video and image processing is based on `FFmpeg <https://www.ffmpeg.org/>`__ and `OpenCV <https://opencv.org/>`__, but Birdwatcher is easier to use for many tasks because its higher-level implementation of functionality.

In addition to video analysis tools, Birdwatcher has high-level functions for behavioral analysis based on such tools, although currently these are limited to movement/location detection of single animals.

Despite its name, Birdwatcher is not only for birds. We also successfully analyzed dog behavior, and it could be used on anything that moves. It is being used in our lab but still under heavy development, and should be considered alpha software.

Code can be found on GitHub: https://github.com/gbeckers/Birdwatcher .

Example notebooks are currently the best introduction on how to use
Birdwatcher. See `jupyter notebook directory <https://github.com/gbeckers/Birdwatcher/tree/master/notebooks>`__.

Documentation can be found at https://birdwatcher.readthedocs.io .

It is developed by Gabriel Beckers and Carien Mol, at Experimental Psychology,
Utrecht University. It is open source, freely available under the `New BSD License
<https://opensource.org/licenses/BSD-3-Clause>`__ terms.


Installation Birdwatcher package
--------------------------------

Birdwatcher officially supports Python 3.13 or higher, but older
Python 3 versions may also work.

**User installation**

#. We recommend using Anaconda for installation. Install Anaconda from https://www.anaconda.com/ .

#. Open Anaconda prompt in terminal.

#. Create new environment for Birdwatcher (name is up to you, in example here 'mybirdwatcher'). We install Jupyter lab and ffmpeg at the same time:

    .. code-block:: bash

      $ conda create -n mybirdwatcher python=3.12 jupyterlab ffmpeg=4.2.2 git

#. Switch to this new environment:

    .. code-block:: bash

      $ conda activate mybirdwatcher

#. Install Birdwatcher:

   Stable latest official release from PyPi:

    .. code-block:: bash

      $ pip install Birdwatcher

   If instead you want the latest version of the git master branch from, use:

    .. code-block:: bash

      $ pip install git+https://github.com/gbeckers/birdwatcher@master


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


Run notebooks tutorial
----------------------

To quickly learn the fundamentals of Birdwatcher, please walk through our notebooks. First, you need to download the notebooks and example videos from github. Then, navigate to the directory of the notebooks and activate the 'mybirdwatcher' environment. Type `jupyter lab` which opens in your browser. You can now open the notebooks and run the tutorial.


Test
----

To run the test suite:

.. code:: python

    >>>import birdwatcher as bw
    >>>bw.test()
    ..................................................
    ----------------------------------------------------------------------
    Ran 50 tests in 75.858s

    OK
    
    <unittest.runner.TextTestResult run=50 errors=0 failures=0>


Documentation
-------------

https://birdwatcher.readthedocs.io

Examples
--------

See `jupyter notebook directory
<https://github.com/gbeckers/Birdwatcher/tree/master/notebooks>`__.

Contributions
-------------
Sita ter Haar and Dylan Minekus helped exploring the application of movement
detection algorithms.

.. |Repo Status| image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active
.. |Github CI Status| image:: https://github.com/gbeckers/Birdwatcher/actions/workflows/python_package.yml/badge.svg
   :target: https://github.com/gbeckers/Birdwatcher/actions/workflows/python_package.yml
.. |PyPi version| image:: https://img.shields.io/badge/pypi-0.5.0-orange.svg
   :target: https://pypi.org/project/birdwatcher/
.. |Docs Status| image:: https://readthedocs.org/projects/birdwatcher/badge/?version=latest
   :target: https://birdwatcher.readthedocs.io/en/latest/
.. |Codecov status| image:: https://codecov.io/gh/gbeckers/Birdwatcher/branch/master/graph/badge.svg?token=829BH0NSVM
   :target: https://codecov.io/gh/gbeckers/Birdwatcher


