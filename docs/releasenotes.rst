Release notes
=============

Commits by Carien Mol and Gabriel Beckers.
    
New submodule `parameterselection` part of module `movementdetection`:
    - `ParameterSelection` class with the results as Pandas DataFrame, and 
    methods to easily view and compare the results of various parameters
    - `apply_all_parameters` and `load_parameterselection` function
    - `choose_parameters` notebook

Other changes:
    - `product_dict` function in utils module
    - some restrictions in what is imported automatically via the init file
    - added more tests
	
Some corrections:
	- switch frameheight/framewidth when calling framecolor or framegray
	- also include first frame when calculating the mean frame in `calc_meanframe`


Version 0.3
-----------

Commits by Carien Mol and Gabriel Beckers.

New methods:
	- `peek_frames` method for Frames for peeking the first frame
	- `show` method for Frames, VideoFileStream and CoordinateArrays
	- `save_nonzero` method for Frames to directly save nonzero pixel 
    coordinates as CoordinateArrays
	- `get_coordmedian` method for CoordinateArrays
	- `edge_detection` method for Frames

Other changes:
	- bug correction: switch frameheight/framewidth when initializing Frames 
    object
	- `find_nonzero` methods now can work with color frames
	- `apply_backgroundsegmenter` method on Frames for background segmentation
	- improved logging of processing steps
	- many improvements docstrings and code consistency
	- added more tests