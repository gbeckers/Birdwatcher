from pathlib import Path
import numpy as np
import tarfile

from darr import RaggedArray, delete_raggedarray

from .videoinput import VideoFile

__all__ = ['CoordinateData']

class CoordinateData(RaggedArray):

    def __init__(self, path):

        super().__init__(path=path)
        md = dict(self.metadata)
        self.width = md['videowidth']
        self.height = md['videoheight']

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
            delete_raggedarray(self)

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
