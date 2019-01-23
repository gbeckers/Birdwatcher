import matplotlib.pyplot as plt
import cv2 as cv

__all__ = ['imshow_frame']

def imshow_frame(frame, fig=None, figsize=None, cmap=None):
    if fig is None:
        w = 14
        h = frame.shape[1] * w / frame.shape[0]
        plt.figure(figsize=(w,h))
    if cmap is None:
        if frame.ndim == 3:
            cmap = None
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        elif frame.ndim ==2:
            cmap = 'gray'
    plt.imshow(frame.astype('uint8'), cmap=cmap)