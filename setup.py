import sys
import versioneer
import setuptools

# we follow https://scientific-python.org/specs/spec-0000/
if sys.version_info < (3,9):
    print("Birdwatcher requires Python 3.9 or higher please upgrade")
    sys.exit(1)

long_description = \
"""|Github CI Status| |PyPi version| |Docs Status| |Repo Status|
|Codecov status|

.. image:: https://raw.githubusercontent.com/gbeckers/Birdwatcher/57e1c452c6ee6d51b70acf52da8a3e316adc097a/docs/images/banner.gif
  :align: center
  :width: 720

Birdwatcher is a Python computer vision library for analyzing animal behavior
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

In addition to video analysis tools, Birdwatcher has high-level functions 
for behavioral analysis based on such tools, although currently these are 
limited to movement/location detection of single animals.


Despite its name, Birdwatcher is not only for birds. We also successfully
analyzed dog behavior, and it could be used on anything that moves. It is
being used in our lab but still under heavy development, and should be
considered alpha software.

Code can be found on GitHub: https://github.com/gbeckers/Birdwatcher .

Documentation can be found at https://birdwatcher.readthedocs.io .

It is developed by Gabriel Beckers and Carien Mol, at Experimental Psychology,
Utrecht University. It is open source, freely available under the `New BSD License
<https://opensource.org/licenses/BSD-3-Clause>`__ terms.

.. |Repo Status| image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active
.. |Github CI Status| image:: https://github.com/gbeckers/Birdwatcher/actions/workflows/python_package.yml/badge.svg
   :target: https://github.com/gbeckers/Birdwatcher/actions/workflows/python_package.yml
.. |PyPi version| image:: https://img.shields.io/badge/pypi-0.4.0-orange.svg
   :target: https://pypi.org/project/birdwatcher/
.. |Docs Status| image:: https://readthedocs.org/projects/birdwatcher/badge/?version=latest
   :target: https://birdwatcher.readthedocs.io/en/latest/
.. |Codecov status| image:: https://codecov.io/gh/gbeckers/Birdwatcher/branch/master/graph/badge.svg?token=829BH0NSVM
   :target: https://codecov.io/gh/gbeckers/Birdwatcher


"""

setuptools.setup(
    name='birdwatcher',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['birdwatcher', 'birdwatcher.movementdetection', 'birdwatcher.tests', 'birdwatcher.testvideos'],
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
