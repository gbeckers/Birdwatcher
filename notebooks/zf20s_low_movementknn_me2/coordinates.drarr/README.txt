Disk-based storage of a ragged array
====================================

This directory is a data store for a numeric ragged array. This can be seen 
as a sequence of subarrays that have the same shape except for one 
dimension. On disk, these subarrays are concatenated along their variable 
dimension. The data can easily be read using the Darr library in Python, 
but if that is not available, they can also be read in other environments. 
To do so, read below.

There are two subdirectories, each containing an array stored in a simple 
format that are easy to read. See the README's in the corresponding 
directories to find out how. The subdirectory 'values' holds the numerical 
data itself, where subarrays are simply appended along their variable length 
dimension (first axis). So the number of dimensions of the values array is 
one less than that of the ragged array. Read the values array first. A 
particular subarray can be then be retrieved if one knows its start and end 
index along the first axis of the values array. These indices (counting from 
0) are stored in a different 2-dimensional array in the subdirectory 
'indices'. The first axis of the index array represents the sequence number 
of the subarray and the second axis (length 2) represents start and end 
indices to be used on the values array.

So to read the n-th subarray, read the nt-h start and end indices 
from the indices array ('starti, endi = indices[n]') and use these to read the 
array data from the values array ('array = values[starti:endi]').


