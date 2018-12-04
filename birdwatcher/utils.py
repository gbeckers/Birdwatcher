import shutil
import tempfile
import itertools
from contextlib import contextmanager

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