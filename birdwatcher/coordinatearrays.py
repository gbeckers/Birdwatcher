"""This module provides objects and functions for coordinate data.
Coordinates are pixel positions (x,y). This is a convenient way of storing
output from image algorithms that determine whether or not a pixel is
categorized as something (e.g. crossing some threshold). Since the number of
pixels per image in a video is different, we use disk-based ragged arrays
from the python library Darr to store them. They can be archived in
compressed form (lzma) when data is large.

"""


from contextlib import contextmanager
from pathlib import Path
import numpy as np
import tarfile

from darr import RaggedArray, delete_raggedarray, create_raggedarray

from ._version import get_versions
from .utils import tempdir

__all__ = ['CoordinateArrays', 'open_archivedcoordinatedata',
           'create_coordarray']

# fixme, should we allow for 3 values: (x, y, val)?
class CoordinateArrays(RaggedArray):

    def __init__(self, path, accessmode='r'):
        super().__init__(path=path, accessmode=accessmode)
        md = dict(self.metadata)
        self.width = md['video_width']
        self.height = md['video_height']


    def get_frame(self, frameno):
        frame = np.zeros((self.height, self.width), dtype=np.bool)
        coords = self[frameno]
        frame[(coords[:, 1], coords[:, 0])] = 1
        return frame

    def archive(self, remove_source=False):
        apath = f'{self.path}.tar.xz'
        with tarfile.open(apath, "w:xz") as tf:
            tf.add(self.path)
        if remove_source:
            self.accessmode = 'r+'
            delete_raggedarray(self)
        return Path(apath)


def create_coordarray(path, videofile,  metadata=None, overwrite=True):

    if metadata is None:
        metadata = {}
    metadata.update(videofile.get_properties(affix='video_'))
    metadata.update({'birdwatcher_version': get_versions()['version']})
    coords = create_raggedarray(path, atom=(2,), metadata=metadata,
                                overwrite=overwrite)
    return CoordinateArrays(coords, accessmode='r+')



@ contextmanager
def open_archivedcoordinatedata(path):
    path = Path(path)
    if not path.suffix == '.xz':
        raise OSError(f'{path} does not seem to be archived coordinate data')

    with tempdir() as dirname:
        tar = tarfile.open(path)
        tar.extractall(path=dirname)
        tar.close()
        p = path.parts[-1].split('.tar.xz')[0]
        yield CoordinateArrays(Path(dirname) / Path(p))


# class CoordinateAnalyis(CoordinateH5Data):
#
#     def __init__(self, coordinatefilepath, videofilepath=None,
#                  coordnode='/pixelcoordinates'):
#         if videofilepath is not None:
#             self.videofile = VideoFile(videofilepath)
#         else:
#             self.videofile = None
#         super().__init__(coordinatefilepath=coordinatefilepath,
#                          coordnode=coordnode)
#
#     def show_frame(self, frameno, includevideo=True, fig=None,
#                    figsize=(18, 18)):
#         import matplotlib.pyplot as plt
#         if fig is None:
#             fig = plt.figure(figsize=figsize)
#         thresh = self.read_frame(frameno)
#         if self.videofile is not None and includevideo:
#             videoframe = self.videofile.get_framebynumber(frameno)
#             videoframe[thresh, 2] = 255
#             thresh = videoframe
#         plt.imshow(thresh)
#
#     def get_coordcount(self, startframeno=None, endframeno=None):
#         coordgen = self.iter_coordinates(startframeno=startframeno,
#                                          endframeno=endframeno)
#         return np.array([c.shape[0] for c in coordgen])
#
#     def get_coordmean(self, startframeno=None, endframeno=None):
#         coordgen = self.iter_coordinates(startframeno=startframeno,
#                                          endframeno=endframeno)
#         return np.array([c.mean(0) for c in coordgen])
