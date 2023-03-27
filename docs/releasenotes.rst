Release notes
=============

Master
------

Commits by Carien Mol and Gabriel Beckers.

New methods:
	- `peek_frames` method for Frames for peeking the first frame
	- `show` method for Frames, VideoFileStream and CoordinateArrays
	- 'save_nonzero' method for Frames to directly save nonzero pixel coordinates as CoordinateArrays
	- 'get_coordmedian' method for CoordinateArrays
	- 'edge_detection' method for Frames

Other changes:
	- bug correction: switch frameheight/framewidth when initializing Frames object
	- `find_nonzero` methods now can work with color frames
	- `apply_backgroundsegmenter` method on Frames for background segmentation
	- improved logging of processing steps
	- many improvements docstrings and code consistency
	- added more tests