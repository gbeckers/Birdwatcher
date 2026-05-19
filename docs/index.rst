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
behavior within a Python scientific computing environment. It aims to get you
up and running quickly when building analysis code or tools for specific
measurements.

Birdwatcher offers both high-level and lower-level tools for behavioral analysis.
Currently, higher level functionality is focused on **single-animal movement
and location detection**. This is based on machine learning (background
subtraction algorithms). Birdwatcher focuses on lower level deterministic
algorithms that yield predictable output on the basis of transformations and
decisions that can be understood by the researcher, and that don't require
the extensive training and testing of neural network models. This makes
Birdwatcher a nice alterative to deep-learning approaches for exploration of
data, and robust measurements that don't require the complexity

Furthermore, to build your own high-level tools, Birdwatcher offers lower-level
functionality such as:

- Easy reading and writing videos as NumPy arrays
- Applying processing algorithms like background subtraction
- Morphological transformations, resizing, and frame annotation
- Writing lower-level data output to disk-based ragged-arrays for futher analyses

`Tutorial notebooks <https://github.com/gbeckers/Birdwatcher/tree/master/notebooks>`__
on Github help to learn using Birdwatcher. There are also shorter :doc:`recipes`
providing hands-on examples.

Birdwatcher is based on `FFmpeg <https://www.ffmpeg.org/>`__ and
`OpenCV <https://opencv.org/>`__.

It is developed by Gabriel Beckers and Carien Mol, at Experimental Psychology,
Utrecht University. It is open source, freely available under the `New BSD License
<https://opensource.org/licenses/BSD-3-Clause>`__ terms. Despite its name,
Birdwatcher is not just for birds! We've used it successfully to analyze dog
behavior and it can work for any moving subject.

Code can be found on GitHub: https://github.com/gbeckers/Birdwatcher .


.. toctree::
   :maxdepth: 3
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
