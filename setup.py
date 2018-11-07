import sys
from distutils.core import setup
import versioneer
import setuptools

if sys.version_info < (3,6):
    print("Birdwatcher requires Python 3.6 or higher please upgrade")
    sys.exit(1)

long_description = \
"""Birdwatcher is a Python computer vision library for the analysis of animal
behavior.

It is being used in our lab but not stable enough yet for general use. More
info will be provided when a first release is appropriate.

It is open source and freely available under the `New BSD License
<https://opensource.org/licenses/BSD-3-Clause>`__ terms.
"""

setup(
    name='birdwatcher',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['birdwatcher'],
    url='https://github.com/gbeckers/birdwatcher',
    license='BSD-3',
    author='Gabriel J.L. Beckers',
    author_email='gabriel@gbeckers.nl',
    description='A Python computer vision library for animal behavior',
    long_description=long_description,
    long_description_content_type="text/markdown",
    requires=['numpy', 'darr', 'opencv'],
    install_requires=['numpy'],
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
