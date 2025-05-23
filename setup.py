import sys
import setuptools

# we follow https://scientific-python.org/specs/spec-0000/
if sys.version_info < (3,10):
    print("Birdwatcher requires Python 3.11 or higher please upgrade")
    sys.exit(1)


setuptools.setup(
    packages=['birdwatcher', 'birdwatcher.movementdetection', 'birdwatcher.tests', 'birdwatcher.testvideos'],
    package_data={'birdwatcher.testvideos': ['*.mp4']},
    include_package_data=True,
)
