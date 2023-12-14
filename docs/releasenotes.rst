Release notes
=============

- bug correction in notebook 5: fix copy and update of settings dictionary when looking at range of parameter values.
- `extract_audio' has more argument options, such as specifying the codec and  channel number that should be extracted.


Version 0.4
-----------

Commits by Carien Mol and Gabriel Beckers.

New submodule `parameterselection`, part of module `movementdetection`:
    - `ParameterSelection` class with the results as Pandas DataFrame, and methods to easily view and compare the results of various parameters.
    - `apply_all_parameters` and `load_parameterselection` function.
    - `parameterselection` notebook.
    
Big change to `movementdetection` module:
    - There is now one high-level function `detect_movement`, in which the type of background subtraction algorithm that should be used can be added as optional parameter.
    - There is an easy-to-use default parameters setting for this function that takes care of many of the pre- and postprocessing steps.
    - There is also a optional `settings` parameters in which you can easily set various processing steps, as wel as the parameters values for the background subtraction algorithm, to be in full control of all settings.

Tutorial with five notebooks:
    - The notebooks of previous versions are modified into a more cohesive tutorial. The first three notebooks demonstrate some basic functionalities of Birdwatcher. The fourth and fifth notebook are specifically designed to apply movement detection on the user's own videos.

Other changes:
    - `product_dict` function in utils module.
    - some restrictions in what is imported automatically via the init file.
    - modified existing tests and added new tests.
	
Some corrections:
	- switch frameheight/framewidth when calling framecolor or framegray.
	- also include first frame when calculating the mean frame in `calc_meanframe`.


Version 0.3
-----------

Commits by Carien Mol and Gabriel Beckers.

New methods:
	- `peek_frames` method for Frames for peeking the first frame
	- `show` method for Frames, VideoFileStream and CoordinateArrays
	- `save_nonzero` method for Frames to directly save nonzero pixel coordinates as CoordinateArrays
	- `get_coordmedian` method for CoordinateArrays
	- `edge_detection` method for Frames

Other changes:
	- bug correction: switch frameheight/framewidth when initializing Frames object
	- `find_nonzero` methods now can work with color frames
	- `apply_backgroundsegmenter` method on Frames for background segmentation
	- improved logging of processing steps
	- many improvements docstrings and code consistency
	- added more tests