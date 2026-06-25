import itertools
import os, sys
import pathlib
import shutil
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Generator


def roi_to_npindex(
    roi: tuple[int, int, int, int] | list[int],
) -> tuple[slice, slice]:
    """Convert a region of interest (roi) to a numpy index tuple."""
    return (slice(roi[0], roi[1]), slice(roi[2], roi[3]))


def datetimestring() -> str:
    """Returns a string of the current date and time."""
    return time.strftime("%Y%m%d%H%M%S")


def product_dict(**kwargs) -> Generator[dict[Any, Any], Any, None]:
    """Generates a Cartesian product of dictionary values."""
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


@contextmanager
def tempdir(dirname=".", keep=False, report=False) -> Generator[pathlib.Path, None, None]:
    """Yields a temporary directory which is removed when context is
    closed."""
    tempdirname = tempfile.mkdtemp(dir=dirname)
    if report:
        print("created tempdir {}".format(tempdirname))
    try:
        yield pathlib.Path(tempdirname)
    finally:
        if not keep:
            shutil.rmtree(tempdirname)
            if report:
                print("removed temp dir {}".format(tempdirname))


def peek_iterable(iterable) -> tuple[Any, Generator[Any, None, None]]:
    """Yields the first item of an iterable plus a chained iterator
    that still includes that first item.

    Raises ValueError if the input iterable is empty.
    """
    gen = (i for i in iterable)
    try:
        first = next(gen)
    except StopIteration:
        raise ValueError("cannot peek into an empty iterable") from None
    return first, itertools.chain([first], gen)


def derive_filepath(filepath: str | pathlib.Path, append_string="", suffix=None, path=None) -> pathlib.Path:
    """Generate a file path based on the name and potentially path of the
    input file path.

    Parameters
    ----------
    filepath : str of pathlib.Path
        Path to file.
    append_string : str, optional
        String to append to file name stem.
    suffix : str, optional
        File extension to use. If None, the same as video file.
    path : str or pathlib.Path, optional
        Path to use. If None, use same path as video file.

    Returns
    -------
    pathlib.Path
        Path derived from video file path.

    """
    filepath = pathlib.Path(filepath)
    stem = filepath.stem
    if suffix is None:
        suffix = filepath.suffix
    filename = f"{stem}_{append_string}{suffix}"
    if path is None:
        dpath = filepath.parent / filename
    else:
        dpath = pathlib.Path(path) / filename
    return dpath


def progress(count: int, total: int, status: str = "") -> None:

    # The MIT License (MIT)
    # Copyright (c) 2016 Vladimir Ignatev
    #
    # Permission is hereby granted, free of charge, to any person obtaining
    # a copy of this software and associated documentation files (the "Software"),
    # to deal in the Software without restriction, including without limitation
    # the rights to use, copy, modify, merge, publish, distribute, sublicense,
    # and/or sell copies of the Software, and to permit persons to whom the Software
    # is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included
    # in all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    # PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    # FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
    # OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    # OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write("[%s] %s%s ...%s\r" % (bar, percents, "%", status))
    sys.stdout.flush()


def print_dirstructure(dirpath: str) -> None:
    """Prints the hierarchical structure of directories, starting at `dirpath`

    Parameters
    ----------
    dirpath : str or Path
        The top-level directory to start at.

    """
    for root, dirs, files in os.walk(dirpath):
        level = root.replace(dirpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            if pathlib.Path(f).is_dir():
                print(f"{subindent}{f}")


def walk_paths(dirpath: str | pathlib.Path, extension=".*") -> Generator[pathlib.Path, None, None]:
    """Walks recursively over contents of `dirpath` and yield contents as
    pathlib Path objects, potentially based on their `extension`.

    Parameters
    ----------
    dirpath : str or Path
        The top-level directory to start at.
    extension : str, optional
        Filter on this extension. The default includes all extensions.

    """
    dirpath = pathlib.Path(dirpath)
    if extension.startswith("."):
        extension = extension[1:]
    for file in dirpath.rglob(f"*.{extension}"):
        yield file
