Birdwatcher
===========

|Repo Status|

Birdwatcher is a Python computer vision library for the analysis of animal
behavior.

It is being used in our lab but not stable enough yet for general use. More
info will be provided when a first release is appropriate.

Code can be found on GitHub: https://github.com/gbeckers/Birdwatcher

It is open source and freely available under the `New BSD License
<https://opensource.org/licenses/BSD-3-Clause>`__ terms.

Installation
------------

As long as there is no official release:

Install dependencies::

    $ pip install darr
    $ pip install opencv-python
    $ pip install opencv-contrib-python

Install Birdwatcher from git repo::

    $ pip install git+https://github.com/gbeckers/birdwatcher


Test
----
To run the test suite:

.. code:: python

    >>>import birdwatcher as bw
    >>>bw.test()
    .
    ----------------------------------------------------------------------
    Ran 1 test in 0.012s

    OK
    <unittest.runner.TextTestResult run=1 errors=0 failures=0>


..  |Repo Status| image:: https://www.repostatus.org/badges/latest/wip.svg
    :alt: Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.
    :target: https://www.repostatus.org/#wip