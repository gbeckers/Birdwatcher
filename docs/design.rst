######
Design
######

.. contents:: :local:

**Overall design considerations**

Birdwatcher provides both low-level and high-level functionality.

The low-level functionality is intended to be sufficiently flexible for users to design their own analysis pipeline for a particular behavioral experiment if this has not yet been implemented at a higher level. At the same time, it prevents users from having to deal with the intricacies of the underlying OpenCV and FFMpeg libraries, which saves considerable time and helps avoid many potential bugs. With only a few lines of code, users can loop over frames in a video file, perform operations or measurements, and save the resulting data in a format suitable for efficient further analysis. A reasonable level of proficiency in scientific Python is required, however.

The high-level functionality is designed so that users can perform useful analyses with minimal coding. Often, only a few lines of code using high-level functions are sufficient. One example is movement detection; see the `tutorial notebook <https://github.com/gbeckers/Birdwatcher/blob/develop/notebooks/5_movementdetection.ipynb>`__ on
GitHub or the shorter :doc:`recipes` in the documentation. Users do not necessarily need advanced Python skills, because the code examples are intended to be self-explanatory and easy to adapt to specific situations.

**Quality and usability considerations**

Birdwatcher is open-source and freely available. Code quality is continuously monitored through an automated testing framework. In addition, we aim to provide thorough documentation for all functions and classes. Users are encouraged to report bugs or other issues using `GitHub's issue tracker <https://github.com/gbeckers/Birdwatcher/issues>`__. Users should therefore expect a well-tested and well-documented tool that can be installed easily by following the provided instructions. The code architecture is designed to facilitate rapid identification and resolution of potential problems, as well as future expansion of functionality.


