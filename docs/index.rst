.. Birdwatcher documentation master file

Birdwatcher
===========

|Github CI Status| |PyPi version| |Docs Status| |Repo Status|
|Codecov status|

.. raw:: html

    <div style="margin: 1em 0;">
      <video
        autoplay
        muted
        loop
        playsinline
        width="100%"
        style="border-radius: 12px;"
      >
        <source src="_static/banner.webm" type="video/webm">
      </video>
    </div>

**Birdwatcher** is a Python computer vision library for analyzing animal
behavior within a scientific computing environment.

It aims to get you up and running quickly when building analysis code or tools
for specific measurements.

Birdwatcher offers both high-level and lower-level tools for behavioral analysis.
Currently, higher level functionality is focused on:

- Single-animal movement and location detection

This is based on machine learning, but not on deep learning, as the specific
goal is that the input to output transformation is deterministic and can be
entirely understood.

To build your own high-level tools, Birdwatcher offers lower-level
functionality such as:

- Easy reading and writing videos as NumPy arrays
- Applying processing algorithms like background subtraction
- Morphological transformations, resizing, and frame annotation
- Writing lower-level data output to disk-based ragged-arrays for futher analyses

Birdwatcher provides an interface to the underlying low-level tools `FFmpeg
<https://www.ffmpeg.org/>`__ and `OpenCV <https://opencv.org/>`__ and
simplifies many tasks in typical animal behavior work.



Despite its name, Birdwatcher is not just for birds! We've used it successfully to analyze dog behavior and it can work for any moving subject.

Code can be found on GitHub: https://github.com/gbeckers/Birdwatcher .

Example notebooks are currently the best introduction on how to use
Birdwatcher. See `jupyter notebook directory <https://github.com/gbeckers/Birdwatcher/tree/master/notebooks>`__.

It is developed by Gabriel Beckers and Carien Mol, at Experimental Psychology,
Utrecht University. It is open source, freely available under the `New BSD License
<https://opensource.org/licenses/BSD-3-Clause>`__ terms.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   recipes
   design
   api
   development
   troubleshooting
   releasenotes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


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
