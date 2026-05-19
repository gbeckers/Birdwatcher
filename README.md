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

[Tutorial notebooks](https://github.com/gbeckers/Birdwatcher/tree/master/notebooks)
on Github help to learn using Birdwatcher. There are also shorter 
[recipes](https://birdwatcher.readthedocs.io/en/develop/recipes.html) providing 
hands-on examples in the [documentation](https://birdwatcher.readthedocs.io/en/develop/).

Birdwatcher is based on [FFmpeg](https://www.ffmpeg.org/) and [OpenCV](https://opencv.org/).

It is developed by Gabriel Beckers and Carien Mol, at Experimental Psychology,
Utrecht University. It is open source, freely available under the [New BSD 
License](https://opensource.org/licenses/BSD-3-Clause) terms. Despite its name,
Birdwatcher is not just for birds! We've used it successfully to analyze dog
behavior and it can work for any moving subject.

Code can be found on GitHub: https://github.com/gbeckers/Birdwatcher .


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



