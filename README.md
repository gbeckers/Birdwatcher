# Birdwatcher

[![Github CI Status](https://github.com/gbeckers/Birdwatcher/actions/workflows/python_package.yml/badge.svg)](https://github.com/gbeckers/Birdwatcher/actions/workflows/python_package.yml)
[![PyPI version](https://img.shields.io/badge/pypi-0.5.0-orange.svg)](https://pypi.org/project/birdwatcher/)
[![Docs Status](https://readthedocs.org/projects/birdwatcher/badge/?version=latest)](https://birdwatcher.readthedocs.io/en/latest/)
[![Repo Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Codecov status](https://codecov.io/gh/gbeckers/Birdwatcher/branch/master/graph/badge.svg?token=829BH0NSVM)](https://codecov.io/gh/gbeckers/Birdwatcher)

<p align="center">
  <img src="https://github.com/gbeckers/Birdwatcher/blob/master/docs/images/banner.gif" width="720">
</p>


**Birdwatcher** is a Python computer vision library for analyzing animal behavior within a scientific computing environment.

It aims to get you up and running quickly when building analysis code or tools for specific measurements. 

Birdwatcher offers both high-level and lower-level tools for behavioral analysis. Currently, higher level functionality is focused on:

- Single-animal movement
- Location detection

Birdwatcher is based on machine learning, but not on deep learning techniques, as the specific goal is that the input to output transformation is entirely understood and predictable.

Further, to build your own high-level tools Birdwatcher offers lower-level video analysis functionality such as:

- Easy reading and writing videos as NumPy arrays
- Applying processing algorithms like background subtraction
- Morphological transformations, resizing, and frame annotation
- Writing lower-level data output to disk-based ragged-arrays for futher analyses

Birdwatche builds upon the underlying low-level tools [FFmpeg](https://www.ffmpeg.org/) and [OpenCV](https://opencv.org/), but Birdwatcher provides an interface that simplifies many tasks in typical animal behavior work.

Despite its name, Birdwatcher is not just for birds! We've used it successfully to analyze dog behavior and it can work for any moving subject.

*Note*: Birdwatcher is actively used in our lab but remains under heavy development. Consider it *alpha software*.

Example notebooks are currently the best introduction on how to use
Birdwatcher. See [jupyter notebook directory](https://github.com/gbeckers/Birdwatcher/tree/master/notebooks).

Documentation can be found at https://birdwatcher.readthedocs.io .

It is developed by Gabriel Beckers and Carien Mol, at Experimental Psychology,
Utrecht University. It is open source, freely available under the 
[New BSD License](https://opensource.org/licenses/BSD-3-Clause) terms.

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
    For development version use:
    
    ```
    pip install git+https://github.com/gbeckers/birdwatcher@master
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



