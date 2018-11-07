import matplotlib.pyplot as plt

__all__ = ['imshow_frame']

def imshow_frame(frame, fig=None, figsize=None):
    if fig is None:
        w = 14
        h = frame.shape[1] * w / frame.shape[0]
        plt.figure(figsize=(w,h))
    plt.imshow(frame.astype('uint8'))