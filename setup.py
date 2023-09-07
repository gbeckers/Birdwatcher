import sys
from distutils.core import setup
import versioneer
import setuptools

# we follow https://scientific-python.org/specs/spec-0000/
if sys.version_info < (3,9):
    print("Birdwatcher requires Python 3.9 or higher please upgrade")
    sys.exit(1)

long_description = \
"""Birdwatcher is a Python computer vision library for analyzing animal behavior
in a Python scientific computing environment.

Birdwatcher should help you getting up and running quickly when building
analysis code or tools for specific measurements. It provides high-level
functionality that is common in video analysis, such as reading and writing
videos into and from numpy arrays, applying processing algorithms such as
background subtraction, morphological transformation, resizing, drawing on
frames etc. Much of the underlying video and image processing is based on
`FFmpeg <https://www.ffmpeg.org/>`__ and `OpenCV <https://opencv.org/>`__,
but Birdwatcher is a lot easier to use for many tasks because its
higher-level implementation of functionality as compared to these tools.

Despite its name, Birdwatcher is not only for birds. We also successfully
analyzed dog behavior, and it could be used on anything that moves. It is
being used in our lab but still under heavy development, and should be
considered alpha software.

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
    packages=['birdwatcher', 'birdwatcher.movementdetection', 'birdwatcher.tests'],
    package_data={'birdwatcher.testvideos': ['*.mp4']},
    include_package_data=True,
    url='https://github.com/gbeckers/birdwatcher',
    license='BSD-3',
    author='Gabriel J.L. Beckers, Carien Mol',
    author_email='gabriel@gbeckers.nl',
    description='A Python computer vision library for animal behavior',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn', 'darr', 
                      'opencv-python', 'opencv-contrib-python'],
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
