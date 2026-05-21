Archiving coordinate results
============================

.. currentmodule:: birdwatcher

Results from movement or location detection prodedures often yield a disk-based :class:`CoordinateArray` data array, that is memory-mapped because it can be very large. Depending on storage space available, it can be useful to *archive* such arrays using the :meth:`CoordinateArray.archive` method, which means that they are compressed and archived into one file. This way, you can reduce storage space by a factor 5-10.

.. code:: python

    >>> import birdwatcher as bw
    >>> ca = bw.CoordinateArrays('myvideo/coords.darr/')
    >>> ca.archive()
    WindowsPath('myvideo/coords.darr.tar.xz')

Birdwatcher does not delete the original uncompressed ``myvideo/coords.darr`` data for your after compression. You should do this yourself.

The archived data cannot be accessed directly from within Python. It first needs uncompressing.

If you want to access your data again at a later time, there are two options. First, you can extract the orginal data from the archive using the :meth:`CoordinateArray.extract_archivedcoordinatedata` method, and use the :class:`CoordinateArray` as normal:

.. code:: python

    >>> ca =  bw.extract_archivedcoordinatedata('myvideo/coords.darr.xz')
    >>> print(ca[99])
    [[492, 393],
     [493, 393],
     [492, 394],
     [493, 394],
     [487, 397],
     [485, 401],
     [483, 402],
     [484, 402],
     [485, 402]], dtype=uint16)

The data will remain uncompressed.

If you only want to access the data once, and keep it archived/compressed after use, you can also use a context manager:

.. code:: python

    >>> with bw.open_archivedcoordinatedata('myvideo/coords.darr.tar.xz') as ca:
            print(ca[99])
    [[492, 393],
     [493, 393],
     [492, 394],
     [493, 394],
     [487, 397],
     [485, 401],
     [483, 402],
     [484, 402],
     [485, 402]], dtype=uint16)

No uncompressed data remains on disk.