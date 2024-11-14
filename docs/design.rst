######
Design
######

.. contents:: :local:

**Overall design considerations**

Birdwatcher provides low-level and high-level functionality.

The low-level functionality is designed to be sufficiently low for users to design their own analysis pipeline for
their particular behavioral experiment that hasn't been implemented at a higher level (yet). But not so low that they
need to concern themselves with the intricacies of the underlying OpenCV and FFMpeg libraries, which saves a lot of
time and avoids many potential bugs. A few lines of code enable you to loop over frames in a video file, perform an
operation or measurement, and save the data in a way that allows for efficient further analyses. A reasonable
proficiency in Python is required though.

The high-level functionality is designed so that users don't have to code much to produce specific data for a useful
analysis. Often, a few lines using high-level functions is sufficient. For example in movement detection. Examples on
how to do this are provided in jupyter notebooks. Users do not need a high level in Python because code and examples are
self-explaining, but it is fair to say that the high-level functionality is probably still too low for people suffering
from programming anxiety.

For people with programming anxiety there is no functionality in Birdwatcher yet, but it would not be difficult to add
a graphical interface for common analysis tasks, such as movement detection. However at the moment a graphical
interface have not been implemented yet.

**Quality and usability considerations**

Birdwatcher is open and free. Quality of the code is be continuously monitored through its automated testing
framework. Furthermore, we aim for functions and classes to be well-documented. Users are invited to report bugs or
other problems using `GitHub's issue tracker<https://github.com/gbeckers/Birdwatcher/issues>`__. That is, users should
expect a tested and documented tool that can be easily installed based on provided instructions. The code architecture
is designed so as to enable rapid identification and fixing of potential problems, or to expand functionality.



