import sys
from distutils.core import setup
import versioneer
import setuptools

if sys.version_info < (3,6):
    print("Birdwatcher requires Python 3.6 or higher please upgrade")
    sys.exit(1)

long_description = \
"""Birdwatcher is a Python computer vision library for the measurement and
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

"""

setup(
    name='birdwatcher',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['birdwatcher', 'birdwatcher.tests'],
    package_data={'birdwatcher.testvideos': ['*.mp4']},
    include_package_data=True,
    url='https://github.com/gbeckers/birdwatcher',
    license='BSD-3',
    author='Gabriel J.L. Beckers',
    author_email='gabriel@gbeckers.nl',
    description='A Python computer vision library for animal behavior',
    long_description=long_description,
    long_description_content_type="text/markdown",
    requires=['numpy', 'darr', 'opencv'],
    install_requires=['numpy', 'darr','matplotlib', 'opencv-python',
                      'opencv-contrib-python'],
    data_files=[("", ["LICENSE"])],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
    ],
)
