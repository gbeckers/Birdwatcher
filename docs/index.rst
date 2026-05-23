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
behavior within a scientific Python workflow. It is designed to help researchers
quickly build analysis pipelines and measurement tools for behavioral data.

Birdwatcher provides both high-level and low-level functionality for behavioral
analysis. Current high-level features focus on **single-animal movement and
location detection** using machine-learning-based background subtraction
algorithms.

A key design goal of Birdwatcher is to emphasize deterministic, interpretable
algorithms that produce predictable results through transformations and decision
steps that can be understood by the researcher. Unlike many deep-learning
approaches, these methods do not require extensive training datasets or model
optimization. This makes Birdwatcher well suited for exploratory analysis,
rapid prototyping, and robust measurements when neural networks would add
unnecessary complexity.

To support custom workflows and higher-level tools, Birdwatcher also includes
lower-level functionality such as:

- Reading and writing videos as NumPy arrays
- Background subtraction and related image-processing algorithms
- Morphological transformations, resizing, and frame annotation
- Storing lower-level output in disk-based ragged arrays for further analysis

To get started, see the `Tutorial notebooks <https://github
.com/gbeckers/Birdwatcher/tree/master/notebooks>`__ on Github. The
`documentation <https://birdwatcher.readthedocs.io/en/latest/>`__
also includes short `recipes <https://birdwatcher.readthedocs.io/en/latest/recipes.html>`__
with hands-on examples.

Birdwatcher is based on `FFmpeg <https://www.ffmpeg.org/>`__ and `OpenCV <https://opencv.org/>`__.

This project is developed by Gabriel Beckers and Carien Mol, at Experimental Psychology,
Utrecht University. Birdwatcher is open source and freely available
under the terms of the terms of the `New BSD License <https://opensource.org/licenses/BSD-3-Clause>`__. Despite its name, Birdwatcher
is not limited to birds — it has also been used  successfully to analyze dog behavior
and can be applied to any moving subject.

Source code: https://github.com/gbeckers/Birdwatcher .


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
