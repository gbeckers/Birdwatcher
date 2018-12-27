import shutil
import tempfile
import itertools
import pathlib
from contextlib import contextmanager

__all__ = ['derive_filepath', 'peek_iterable']

@contextmanager
def tempdir(dirname='.', keep=False, report=False):
    """Yields a temporary directory which is removed when context is closed."""
    try:
        tempdirname = tempfile.mkdtemp(dir=dirname)
        if report:
            print('created tempdir {}'.format(tempdirname))
        yield tempdirname
    except:
        raise
    finally:
        if not keep:
            shutil.rmtree(tempdirname)
            if report:
                print('removed temp dir {}'.format(tempdirname))

def peek_iterable(iterable):
    gen = (i for i in iterable)
    first = next(gen)
    return first, itertools.chain([first], gen)


def derive_filepath(filepath, append_string='', suffix=None, path=None):
    """Generate a file path based on the name and potentially path of the
    input file path.

    Parameters
    ----------
    filepath: str of pathlib.Path
            Path to file.
    append_string: str
        String to append to file name stem. Default: ''.
    suffix: str or None
        File extension to use. If None, the same as video file.
    path: str or pathlib.Path or None
        Path to use. If None use same path as video file.

    Returns
    -------
    pathlib.Path
        Path derived from video file path.

    """
    stem = filepath.stem
    if suffix is None:
        suffix = filepath.suffix
    filename = f'{stem}_{append_string}{suffix}'
    if path is None:
        dpath = filepath.parent / filename
    else:
        dpath = pathlib.Path(path) / filename
    return dpath