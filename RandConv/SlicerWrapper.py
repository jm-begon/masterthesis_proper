# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : May 03 2014
"""

"""
__all__ = ["SlicerWrapper"]

from copy import copy


class SlicerWrapper:

    def __init__(self, wrapped, grpSize):
        self._wrapped = wrapped
        self._nb = grpSize

    def __len__(self):
        try:
            l = len(self._wrapped)//self._nb
        except TypeError:
            l = self._wrapped.shape[0]//self._nb
        return l

    def __iter__(self):
        return SlicerWrapperIterator(self)

    def get(self, index):
        return self._wrapped[(index*self._nb):((index+1)*self._nb)]

    def __getitem__(self, index):
        #If the index is a slice, we return a clone of this object with
        # the sliced pair containers
        if isinstance(index, slice):
            clone = copy(self)
            start = index.start*self._nb
            end = index.stop*self._nb
            clone._wrapped = self._wrapped[start:end]
            return clone
        #If it is a real index (int), we return the corresponding object
        else:
            return self.get(index)

    def getContent(self):
        return self._wrapped


#==============SlicerWrapperIterator Iterator===============
class SlicerWrapperIterator:
    """
    ==============
    SlicerWrapperIterator
    ==============
    An iterator for :class:`SlicerWrapper`
    """
    def __init__(self, ImageBuffer):
        self._buffer = ImageBuffer
        self._index = 0

    def __iter__(self):
        return self

    def next(self):
        if self._index >= len(self._buffer):
            raise StopIteration
        else:
            index = self._index
            self._index += 1
            return self._buffer.get(index)
