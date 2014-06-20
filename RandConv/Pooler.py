# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Apr 03 2014
"""
A set of classes which realise spatial pooling
"""
from abc import ABCMeta, abstractmethod
import numpy as np

__all__ = ["Pooler", "MultiPooler", "IdentityPooler", "ConvolutionalPooler",
           "ConvMaxPooler", "ConvMinPooler", "ConvAvgPooler",
           "MorphOpeningPooler", "MorphClosingPooler",
           "MorphErosionGradientPooler", "MorphDilationGradientPooler",
           "MorphGradientPooler", "MorphLaplacianPooler",
           "MorphBlackHatPooler", "MorphWhiteHatPooler"]


class Pooler:
    """
    ======
    Pooler
    ======
    A :class:`Pooler` performs spatial pooling of a 2D or 3D numpy array.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def pool(self, npArray):
        """
        Aggregate the `npArray`

        Parameters
        -----------
        npArray : 2D or 3D numpy array
            The array to aggregate

        Return
        -------
        aggregated : 2D or 3D numpy array (depending on `npArray`)
            The aggregated array
        """
        pass

    def __call__(self, npArray):
        """
        Delegate to meth:`pool` method
        """
        return self.pool(npArray)


class MultiPooler:
    """
    ===========
    MultiPooler
    ===========
    A :class:`MultiPooler` performs several pooling and returns an iterable
    of results
    """
    def __init__(self, poolerList=[]):
        """
        Construct a :class:`MultiPooler` instance

        Parameters
        ----------
        poolerList : iterable of :class:`Pooler` (default : [])
            The individual :class:`Pooler`s
        """
        self._poolers = poolerList

    def multipool(self, npArray):
        """
        Apply the :class:`Pooler`s to the array

        Parameters
        -----------
        npArray : 2D or 3D numpy array
            The array to aggregate

        Return
        -------
        ls : an iterable of 2D or 3D numpy array (depending on `npArray`)
            The results of the individual :class:`Pooler`s (in the same order)
        """
        if len(self._poolers) == 0:
            return [npArray]
        ls = []
        for pooler in self._poolers:
            ls.append(pooler.pool(npArray))

        return ls

    def __len__(self):
        return len(self._poolers)


class ChainPooler(Pooler):
    """
    ===========
    ChainPooler
    ===========
    A class:`ChainPooler` instance chains several class:`Pooler` instances.
    From a list of class:`Pooler` ls = [p1, p2,..., pn] and an array arr,
    returns : pn.pool(...p2.pool(p1.pool(arr))).
    class:`Pooler`s are applied in the order return by the iterator
    """
    def __init__(self, poolers):
        """
        Parameters
        ----------
        poolers : iterable of class:`Pooler`
            The list of poolers to use
        """
        self._poolers = poolers

    def pool(self, npArray):
        tmp = npArray
        for pooler in self._poolers:
            tmp = pooler.pool(tmp)
        return tmp


class IdentityPooler(Pooler):
    """
    ==============
    IdentityPooler
    ==============
    A :class:`IdentityPooler` return the array untouched (no spatial pooling)
    """

    def pool(self, npArray):
        return npArray


class OutsideRange:
    """
    ============
    OutsideRange
    ============
    An iterable for outside-of-the-2D-array point
    """
    def __init__(self, cMin, cMax, up, down, rowInfset, rowOffset,
                 colInset, colOffset):
        self.cMin = cMin
        self.cMax = cMax
        self.up = up
        self.down = down
        self.rowInfset = rowInfset
        self.rowOffset = rowOffset
        self.colInset = colInset
        self.colOffset = colOffset

    def __iter__(self):
        offCoord = []
         #-- row inf
        for row in self.rowInfset:
            for col in xrange(self.cMin, self.cMax+1):
                offCoord.append((row, col))
        #-- row sup
        for row in self.rowOffset:
            for col in xrange(self.cMin, self.cMax+1):
                offCoord.append((row, col))
        #-- col left
        for col in self.colInfset:
            for row in xrange(self.up, self.down):
                offCoord.append((row, col))
        #-- col right
        for col in self.colOffset:
            for row in xrange(self.up, self.down):
                offCoord.append((row, col))
        return offCoord


class ConvolutionalPooler(Pooler):
    """
    ===================
    ConvolutionalPooler
    ===================
    A class:`ConvolutionalPooler` works by using a convolutional window in
    which it uses a possibly non-linear function.

    Note : pooling function
    -----------------------
    The pooling function must have the following signature :
        res = function(subArray, corners, lsOffCoords, originalArray)
    where
        subArray : a 2D or 3D numpy subarray
            The subarray to process
        corners : tuple = (up, left, down, right)
            The index of the corner :
            subArray = orginalArray[up:down+1, left:right+1]
        lsOffCoords : a list of tuples (x, y)
            The list of points which fall outside of the originalArray
        originalArray: 2D or 3D numpy array
            The original array
        res : number or list of numbers
            The result of the pooling
    """

    def __init__(self, function, height, width):
        """
        Construc a class:`ConvolutionalPooler` instance

        Parameters
        ----------
        function : callable
            A function as mentionned in the class description
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        self._function = function
        self._windowHalfHeight = height//2
        self._windowHalfWidth = width//2

    def crop(self, npArray, row, col):
        """
        Crop, base on this instance window, the array along the first two
        coordinates around the supplied point

        Parameters
        ----------
        npArray : 2D or 3D numpy array
            The array to crop
        row : int > 0
            The row of the point around which to crop
        col : int > 0
            The column of the point around which to crop

        Return
        ------
        tuple = (subArray, lsOffCoords, corners)
        subArray : a 2D or 3D numpy subarray
            The subarray to process
        corners : tuple = (up, left, down, right)
            The index of the corner :
            subArray = npArray[up:down+1, left:right+1]
        lsOffCoords : a list of tuples (x, y)
            The list of points which fall outside of npArray
        """
        npHeight = npArray.shape[0]
        npWidth = npArray.shape[1]
        rMin = row - self._windowHalfHeight
        rMax = row + self._windowHalfHeight
        cMin = col - self._windowHalfWidth
        cMax = col + self._windowHalfWidth
        #Lists of offset index
        rowInfset = []
        rowOffset = []
        colInfset = []
        colOffset = []
        #Cropping coordinate
        u = rMin
        d = rMax + 1  # Inclusive
        l = cMin
        r = cMax + 1  # Inclusive
        if u < 0:
            rowInfset = xrange(rMin, 0)
            u = 0
        if l < 0:
            colInfset = xrange(cMin, 0)
            l = 0
        if d > npHeight:
            rowOffset = xrange(npHeight, d)
            d = npHeight
        if r > npWidth:
            colOffset = xrange(npWidth, r)
            r = npWidth
        #Adding offset coordinates
        offCoord = OutsideRange(cMin, cMax, u, d, rowInfset, rowOffset,
                                colInfset, colOffset)

        return npArray[u:d, l:r], (u, l, d-1, r-1), offCoord,

    def pool(self, npArray):
        """
        Pool the given array according to the instance parameters :
        Apply the instance function on a moving window whose size is determined
        by the instance parameters along the given array

        Parameters
        ----------
        npArray : 2D or 3D numpy array
            The array to pool

        Return
        ------
        result : numpy array
            result depends on the instance parameters
        """
        height, width = npArray.shape[0], npArray.shape[1]
        result = [[0]*width for _ in xrange(height)]
        for row in xrange(height):
            for col in xrange(width):
                subArray, corners, lsOffCoords = self.crop(npArray, row, col)
                result[row][col] = self._function(subArray, corners,
                                                  lsOffCoords, npArray)
        return np.array(result)


class ConvMaxPooler(ConvolutionalPooler):
    """
    =============
    ConvMaxPooler
    =============
    The class:`ConvMaxPooler` replace each pixel (for each color separately)
    by the maximum of the neigborhood
    """
    def __init__(self, height, width):
        """
        Construc a class:`ConvMaxPooler` instance

        Parameters
        ----------
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        ConvolutionalPooler.__init__(self, self._max, height, width)

    @classmethod
    def _max(cls, subArray, corners, lsOffCoords, originalArray):
        """
        Extract the maximum of the subarray in each 3rd dimension and
        return a list of the results
        """
        if (len(subArray.shape)) == 2:
            return subArray.max()
        else:  # shape==3
            vs = []
            for depth in range(subArray.shape[2]):
                vs.append((subArray[:, :, depth]).max())
            return vs


class ConvMinPooler(ConvolutionalPooler):
    """
    =============
    ConvMinPooler
    =============
    The class:`ConvMinPooler` replace each pixel (for each color separately)
    by the maximum of the neigborhood
    """
    def __init__(self, height, width):
        """
        Construc a class:`ConvMinPooler` instance

        Parameters
        ----------
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        ConvolutionalPooler.__init__(self, self._min, height, width)

    @classmethod
    def _min(cls, subArray, corners, lsOffCoords, originalArray):
        """
        Extract the minimum of the subarray in each 3rd dimension and
        return a list of the results
        """
        if (len(subArray.shape)) == 2:
            return subArray.min()
        else:  # shape==3
            vs = []
            for depth in range(subArray.shape[2]):
                vs.append((subArray[:, :, depth]).min())
            return vs


class ConvAvgPooler(ConvolutionalPooler):
    """
    =============
    ConvAvgPooler
    =============
    The class:`ConvAvgPooler` replace each pixel (for each color separately)
    by the maximum of the neigborhood
    """
    def __init__(self, height, width):
        """
        Construc a class:`ConvAvgPooler` instance

        Parameters
        ----------
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        ConvolutionalPooler.__init__(self, self._mean, height, width)

    @classmethod
    def _mean(cls, subArray, corners, lsOffCoords, originalArray):
        """
        Extract the average of the subarray in each 3rd dimension and
        return a list of the results
        """
        if (len(subArray.shape)) == 2:
            return subArray.mean()
        else:  # shape==3
            vs = []
            for depth in range(subArray.shape[2]):
                vs.append((subArray[:, :, depth]).mean())
            return vs


class MorphOpeningPooler(ChainPooler):

    def __init__(self, height, width):
        erosion = ConvMinPooler(height, width)
        dilation = ConvMaxPooler(height, width)
        ChainPooler.__init__(self, [erosion, dilation])


class MorphClosingPooler(ChainPooler):

    def __init__(self, height, width):
        erosion = ConvMinPooler(height, width)
        dilation = ConvMaxPooler(height, width)
        ChainPooler.__init__(self, [dilation, erosion])


class MorphErosionGradientPooler(Pooler):

    def __init__(self, height, width):
        self._erosion = ConvMinPooler(height, width)

    def pool(self, npArray):
        eroded = self._erosion.pool(npArray)
        return npArray - eroded


class MorphDilationGradientPooler(Pooler):

    def __init__(self, height, width):
        self._dilation = ConvMaxPooler(height, width)

    def pool(self, npArray):
        dilated = self._dilation.pool(npArray)
        return dilated - npArray


class MorphGradientPooler(Pooler):

    def __init__(self, height, width):
        self._dilation = ConvMaxPooler(height, width)
        self._erosion = ConvMinPooler(height, width)

    def pool(self, npArray):
        dilated = self._dilation.pool(npArray)
        eroded = self._erosion.pool(npArray)
        morphEroGrad = npArray - eroded
        morphDilGrad = dilated - npArray
        return morphEroGrad + morphDilGrad


class MorphLaplacianPooler(Pooler):

    def __init__(self, height, width):
        self._dilation = ConvMaxPooler(height, width)
        self._erosion = ConvMinPooler(height, width)

    def pool(self, npArray):
        dilated = self._dilation.pool(npArray)
        eroded = self._erosion.pool(npArray)
        morphEroGrad = npArray - eroded
        morphDilGrad = dilated - npArray
        return morphDilGrad - morphEroGrad


class MorphWhiteHatPooler(Pooler):

    def __init__(self, height, width):
        self._opening = MorphOpeningPooler(height, width)

    def pool(self, npArray):
        opened = self._opening.pool(npArray)
        return npArray - opened


class MorphBlackHatPooler(Pooler):

    def __init__(self, height, width):
        self._closing = MorphClosingPooler(height, width)

    def pool(self, npArray):
        closed = self._closing.pool(npArray)
        return closed - npArray


if __name__ == "__main__":
    from PIL import Image
    import pylab as pl

    def imshow(arr, title=None, interpolation="none"):
        pl.figure()
        pl.imshow(arr, cmap="gray", interpolation=interpolation)
        pl.colorbar()
        if title is not None:
            pl.title(title)

    lena = np.array(Image.open("lena.jpg"))[:, :, 0]

    imshow(lena, "lena")
    print "lena", lena.min(), lena.mean(), lena.max()

    height = 3
    width = 3

    avgPooler = ConvAvgPooler(height, width)
    minPooler = ConvMinPooler(height, width)
    opPooler = MorphOpeningPooler(height, width)
    maxPooler = ConvMaxPooler(height, width)
    clsPooler = MorphClosingPooler(height, width)

    avgLena = avgPooler.pool(lena)
    imshow(avgLena, "avgLena")
    print "avgLena", avgLena.min(), avgLena.mean(), avgLena.max()

    minLena = minPooler.pool(lena)
    imshow(minLena, "minLena")
    print "minLena", minLena.min(), minLena.mean(), minLena.max()

    opnLena = opPooler.pool(lena)
    imshow(opnLena, "opnLena")
    print "opnLena", opnLena.min(), opnLena.mean(), opnLena.max()

    maxLena = maxPooler.pool(lena)
    imshow(maxLena, "maxLena")
    print "maxLena", maxLena.min(), maxLena.mean(), maxLena.max()

    clsLena = clsPooler.pool(lena)
    imshow(clsLena, "clsLena")
    print "clsLena", clsLena.min(), clsLena.mean(), clsLena.max()

    geLena = lena - minLena
    imshow(geLena, "Erosion gradient")
    print "geLena", geLena.min(), geLena.mean(), geLena.max()

    gdLena = maxLena - lena
    imshow(gdLena, "Dilatation gradient")
    print "gdLena", gdLena.min(), gdLena.mean(), gdLena.max()

    mgLena = geLena + gdLena
    imshow(mgLena, "Morphological gradient")
    print "mgLena", mgLena.min(), mgLena.mean(), mgLena.max()

    lapLena = gdLena - geLena
    imshow(lapLena, "Morphological Laplacian")
    print "lapLena", lapLena.min(), lapLena.mean(), lapLena.max()

    whLean = lena - opnLena
    imshow(whLean, "White top-hat")
    print "whLean", whLean.min(), whLean.mean(), whLean.max()

    bhLena = clsLena - lena
    imshow(bhLena, "Black top-hat")
    print "bhLena", bhLena.min(), bhLena.mean(), bhLena.max()











