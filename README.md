# Birdwatcher

[![Github CI Status](https://github.com/gbeckers/Birdwatcher/actions/workflows/python_package.yml/badge.svg)](https://github.com/gbeckers/Birdwatcher/actions/workflows/python_package.yml)
[![PyPI version](https://img.shields.io/badge/pypi-0.5.0-orange.svg)](https://pypi.org/project/birdwatcher/)
[![Docs Status](https://readthedocs.org/projects/birdwatcher/badge/?version=latest)](https://birdwatcher.readthedocs.io/en/latest/)
[![Repo Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Codecov status](https://codecov.io/gh/gbeckers/Birdwatcher/branch/master/graph/badge.svg?token=829BH0NSVM)](https://codecov.io/gh/gbeckers/Birdwatcher)

<p align="left">
  <img src="https://github.com/gbeckers/Birdwatcher/blob/develop/videos/banner.gif">
</p>

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

To get started, see the [Tutorial notebooks](https://github.com/gbeckers/Birdwatcher/tree/master/notebooks)
on Github. The [documentation](https://birdwatcher.readthedocs.io/en/latest/) 
also includes short [recipes](https://birdwatcher.readthedocs.io/en/latest/recipes.html) with 
hands-on examples.

Birdwatcher is based on [FFmpeg](https://www.ffmpeg.org/) and [OpenCV](https://opencv.org/).

This project is developed by Gabriel Beckers and Carien Mol, at Experimental Psychology,
Utrecht University. Birdwatcher is open source and freely available
under the terms of the terms of the [New BSD 
License](https://opensource.org/licenses/BSD-3-Clause). Despite its name, Birdwatcher 
is not limited to birds — it has also been used  successfully to analyze dog behavior 
and can be applied to any moving subject.

Source code: https://github.com/gbeckers/Birdwatcher .


## Installation Birdwatcher package

Birdwatcher officially supports Python 3.12 or higher, but older
Python 3 versions may also work.

**User installation**

1. We recommend using Anaconda for installation. Install Anaconda from https://www.anaconda.com/ .

2. Open Anaconda prompt in terminal.

3. Create new environment for Birdwatcher (name is up to you, in example here 'mybirdwatcher'). We install some other packages at the same time:

    ```
    conda create -n mybirdwatcher python=3.14 git=2.51 ffmpeg=8.0.1 numpy=2.4.2 matplotlib=3.10.8 pandas=3.0.0 seaborn=0.13.2
    ```
4. Switch to this new environment:

    ```
    conda activate mybirdwatcher
    ```

5. Install jupyter from conda-forge (currently conda one has a problem).

    ```
    conda install -c conda-forge jupyter
    ```

6. Install Birdwatcher:

   Stable latest official release from PyPi:

   ```
   pip install Birdwatcher
   ```

   If instead you want the latest version of the git master branch, use:

    ```
    pip install git+https://github.com/gbeckers/birdwatcher@master
    ```
    For the development version use:
    
    ```
    pip install git+https://github.com/gbeckers/birdwatcher@develop
    ```

## Run notebooks tutorial


To quickly learn the fundamentals of Birdwatcher, please walk through our notebooks.
See [jupyter notebook directory](https://github.com/gbeckers/Birdwatcher/tree/master/notebooks).

## Documentation

[Documentation on readthedocs](https://birdwatcher.readthedocs.io)

## Examples

See [jupyter notebook directory](https://github.com/gbeckers/Birdwatcher/tree/master/notebooks).

## Contributions

Sita ter Haar and Dylan Minekus helped exploring the application of movement
detection algorithms.



