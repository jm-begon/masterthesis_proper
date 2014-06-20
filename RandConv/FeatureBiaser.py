# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Apr 06 2014
"""
A class which can put more emphasis on some features by articially multipling
the number of columns associated to a given variable
"""
import numpy as np
from copy import copy


class NpView:
    def __init__(self, npArray, biasList):
        self._data = npArray
        self.ndim = npArray.ndim
        self.dtype = npArray.dtype
        self.itemsize = npArray.itemsize
        length = len(npArray)
        #Constructing the inverse look up table (from data to this instance)
        lutInv = [1]*length

        for index, factor in biasList:
            lutInv[index] = factor

        nbNewFeatures = sum(lutInv)
        self._lut = [0]*nbNewFeatures

        lutIndex = 0
        for index in range(len(lutInv)):
            for inc in range(lutInv[index]):
                self._lut[lutIndex] = index
                lutIndex += 1

        self.pack()

    def pack(self):
        ls = list(self._data.shape)
        ls[0] = len(self._lut)
        self.shape = tuple(ls)
        self.size = np.prod(self.shape)

    def get(self, index):
        return self._data[self._lut[index]]

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        return ViewIterator(self)

    def __getitem__(self, index):
        try:
            index[0]
            #Worked ? We have got a sequence
            if isinstance(index, list):
                #Selection list
                clone = copy(self)
                length = len(index)
                newLut = [0]*length
                for i in xrange(length):
                    newLut[i] = index[i]
                clone._lut = newLut
                clone.pack()
                return clone
            else:
                #Slicing or selection recursively
                tmp = self[index[0]]
                if len(index) == 2:
                    return tmp[index[1]]
                else:
                    return tmp[index[1:]]
        except TypeError:
            try:
                index + 1
                #Worked ? We have got an integer --> index
                return self.get(index)
            except TypeError:
                #We should have a slice
                clone = copy(self)
                newLut = self._lut[index]
                clone._lut = newLut
                clone.pack()
                return clone

    def __str__(self):
        rep = ["["]
        for v in self:
            rep.append(str(v))
            rep.append(", ")
        rep = rep[:-1]
        rep.append("]")
        return "".join(rep)


class ViewIterator:
    """
    ==============
    ViewIterator
    ==============
    An iterator for :class:`NpVeiw`
    """
    def __init__(self, view):
        self._view = view
        self._index = 0

    def __iter__(self):
        return self

    def next(self):
        if self._index >= len(self._view):
            raise StopIteration
        else:
            index = self._index
            self._index += 1
            return self._view.get(index)


class FeatureBiaser:
    """
    =============
    FeatureBiaser
    =============
    A :class:`FeatureBiaser` is a view of a 2D numpy array biasing some column
    by artificially multipling them
    """

    def __init__(self, npArray2D, biasList):
        """
        Construct a :class:`FeatureBiaser` instance

        Parameters
        ----------
        npArray2D : 2D numpy array
            The learning array
        biasList : iterable of pairs (index, factor)
            The biasing list
            index : int >= 0
                The index of a feature to bias
            factor : int >= 0
                The multiplying factor for the given column.
                if = 0 : the feature is ignore in the view
                if = 1 : the feature is simply present (same as not having the
                pair in the param:`biasList`)
                if > 1 : the feature is present multiple times in the view
        """
        self._data = npArray2D
        self.ndim = 2
        self.dtype = npArray2D.dtype
        self.itemsize = npArray2D.itemsize
        height, width = npArray2D.shape[0], npArray2D.shape[1]

        #Constructing the inverse look up table (from data to this instance)
        lutInv = [1]*width

        for index, factor in biasList:
            lutInv[index] = factor

        nbNewFeatures = sum(lutInv)
        self._lut = [0]*nbNewFeatures

        lutIndex = 0
        for index in range(len(lutInv)):
            for inc in range(lutInv[index]):
                self._lut[lutIndex] = index
                lutIndex += 1

        self.pack()

    def pack(self):
        ls = list(self._data.shape)
        ls[1] = len(self._lut)
        self.shape = tuple(ls)
        self.size = np.prod(self.shape)

    def get(self, row):
        view = NpView(self._data[row], [])
        view._lut = self._lut
        view.pack()
        return view

    def getElement(self, row, col):
        """
        Return the element at the specified position

        Parameters
        ----------
        row : int >= 0
            The row of the element
        col : int >=0
            The column of the element

        Return
        ------
        el :
            The element
        """
        return self._data[row][self._lut[col]]

    def __str__(self):
        rep = []
        for row in xrange(self.shape[0]):
            for col in xrange(self.shape[1]):
                rep.append(str(self.getElement(row, col)))
                rep.append("\t")
            rep.append("\n")
        return "".join(rep[:-1])

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return ViewIterator(self)

    def __getitem__(self, index):
        try:
            index[0]
            #Worked ? We have got a sequence
            if isinstance(index, list):
                #Selection list
                clone = copy(self)
                clone._data = self._data[index]
                clone.pack()
                return clone
            else:
                #Slicing or selection recursively
                ind1, ind2 = index
                return self[ind1][ind2]
        except TypeError:
            try:
                index + 1
                #Worked ? We have got an integer --> index
                return self.get(index)
            except TypeError:
                #We should have a slice
                clone = copy(self)
                clone._data = self._data[index]
                clone.pack()
                return clone

    def asContiguousArray(self):
        tmp = np.zeros(self.shape)
        height, width = self.shape
        for row in xrange(height):
            for col in xrange(width):
                tmp[row][col] = self.getElement(row, col)
        return tmp


if __name__ == "__main__":
    from sklearn.ensemble import ExtraTreesClassifier
    classif = ExtraTreesClassifier()

    nbObj = 10
    nbFeat = 7
    nbClass = 4
    Xl = np.random.rand(nbObj, nbFeat)
    yl = np.random.randint(0, nbClass-1, nbObj)

    Xlv = FeatureBiaser(Xl, [(0, 3), (1, 3)])
    Xlv = Xlv.asContiguousArray()
    print Xl.shape, Xlv.shape

    classif.fit(Xlv, yl)

    Xt = np.random.rand(nbObj, nbFeat)
    Xtv = FeatureBiaser(Xt, [(0, 3), (1, 3)])
    Xtv = Xtv.asContiguousArray()
    print Xt.shape, Xtv.shape

    classif.predict(Xtv)
