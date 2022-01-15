import warnings
try:
    import pypylon
except ImportError:
    warnings.warn("pypylon library not found, no recording functionality "
                  "available")

from pathlib import Path
from .frameprocessing import Frames
from .video import VideoFileStream
from .utils import datetimestring

__all__ = ['record_pylon']

# useful info: https://www.pythonforthelab.com/blog/getting-started-with-basler-cameras/

def record_pylon(filepath, duration=5.0, framerate=25):

    cam = pypylon.pylon.InstantCamera(pypylon.pylon.TlFactory.GetInstance().CreateFirstDevice())
    cam.Open()
    #cam.AcquisitionMode.SetValue('Continuous')
    minLowerLimit = cam.AutoGainLowerLimit.GetMin()
    maxUpperLimit = cam.AutoGainUpperLimit.GetMax()
    cam.AutoGainLowerLimit.SetValue(minLowerLimit)
    cam.AutoGainUpperLimit.SetValue(maxUpperLimit)
    cam.GainAuto.SetValue('Continuous')
    #cam.ExposureAuto.SetValue('Continuous')
    exposuretime = (1/framerate)*1e6
    cam.ExposureTime.SetValue(exposuretime)
    cam.DigitalShift.SetValue(1)
    cam.Gamma.SetValue(0.5)
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(framerate)
    actualframerate = cam.ResultingFrameRate.GetValue()
    nframes = int(round(actualframerate * duration))

    filepath = Path(filepath)
    filepath = Path(filepath.stem + '_' + datetimestring() + filepath.suffix)
    def start():
        #cam.PixelFormat = "Mono8"
        cam.StartGrabbing(pypylon.pylon.GrabStrategy_OneByOne)
        i = 0
        while cam.IsGrabbing():
            res = cam.RetrieveResult(1000)
            if res.GrabSucceeded():
                yield res.Array
                # print(res.BlockID)
            res.Release()
            i += 1
            if i == nframes:
                cam.StopGrabbing()
                break

    Frames(start()).tovideo(filepath, framerate=actualframerate, pixfmt='gray')
    return VideoFileStream(filepath)