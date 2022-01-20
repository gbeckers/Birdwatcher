.. Birdwatcher documentation master file, created by
   sphinx-quickstart on Sat Dec 15 07:58:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Birdwatcher
===========

|Github CI Status| |Appveyor Status| |Docs Status| |Repo Status|

Birdwatcher is a Python computer vision library for the measurement and
analysis of animal behavior.

The purpose of this library is to provide base functionality for analysing
animal behavior in a Python scientific computing environment. It
is not intended as a specialized final product to analyse specific behaviors
of specific animals, but rather to facilitate working efficiently with
video data in a scientific Python environment and to apply computer vision
algorithms. Birdwatcher should help you getting up and running quickly when
building your own specific analysis code or measurement tools. It provides
functionality that is common in video analysis, such as reading and writing
videos into and from numpy arrays, applying processing algorithms such as
background subtraction, morphological transformation, resizing, drawing on
frames etc. Much of the underlying video and image processing is based on
`FFmpeg <https://www.ffmpeg.org/>`__ and `OpenCV <https://opencv.org/>`__,
but Birdwatcher is a lot easier to use for many tasks.

Despite its name, Birdwatcher is not only for birds. We also successfully
analyze dog behavior, and it could be used on anything that moves. It is
being used in our lab but not stable enough yet for general use. More info
will be provided when a first release is appropriate.

Code can be found on GitHub: https://github.com/gbeckers/Birdwatcher .

Documentation can be found at https://birdwatcher.readthedocs.io .

It is developed by Gabriel Beckers and Carien Mol, at Experimental Psychology,
Utrecht University. It is open source, freely available under the `New BSD License
<https://opensource.org/licenses/BSD-3-Clause>`__ terms.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   design
   api
   testing

Examples
--------

See `jupyter notebook directory on github
<https://github .com/gbeckers/Birdwatcher/tree/master/notebooks>`__ .


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |Repo Status| image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active
.. |Github CI Status| image:: https://github.com/gbeckers/Birdwatcher/actions/workflows/python_package.yml/badge.svg
   :target: https://github.com/gbeckers/Darr/actions/workflows/python_package.yml
.. |Appveyor Status| image:: https://ci.appveyor.com/api/projects/status/github/gbeckers/darr?svg=true
   :target: https://ci.appveyor.com/project/gbeckers/birdwatcher
.. |Docs Status| image:: https://readthedocs.org/projects/birdwatcher/badge/?version=latest
   :target: https://birdwatcher.readthedocs.io/en/latest/

