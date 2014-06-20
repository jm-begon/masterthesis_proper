# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 27 2014
"""
A feature compression scheme
"""
from abc import ABCMeta, abstractmethod
import random
import numpy as np

#from sklearn.decomposition import PCA


class Compressor:
    """
    ==========
    Compressor
    ==========
    A :class:`Compressor` is an object which reduces the number of features.
    The :class:`Compressor` must specify how it compresses the data

    Attributes
    ----------
    n_components : the number of features that the compressor produce
    """

    __metaclass__ = ABCMeta

    def __init__(self, n_components):
        """
        Create a :class:`Compressor` instance

        Parameters
        ----------
        n_components : int > 0
            The number of components to keep
        """
        self.n_components = n_components

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the data

        Parameters
        ----------
        X : 2D numpy array
            The feature array
        y : iterable of int
            The label vector

        Return
        ------
        self : :class:`Compressor`
            This object
        """
        return self

    @abstractmethod
    def transform(self, X):
        """
        Transform the data

        Parameters
        ----------
        X : 2D numpy array
            The feature array

        Return
        ------
        X2 : 2D numpy array
            The transformed feature array
        """
        pass

    def fit_transform(self, X, y):
        """
        Fit and transform the data

        Parameters
        ----------
        X : 2D numpy array
            The feature array
        y : iterable of int
            The label vector

        Return
        ------
        X2 : 2D numpy array
            The transformed feature array
        """
        return self.fit(X, y).transform(X)


class CompressorFactory:
    """
    =================
    CompressorFactory
    =================
    A factory class which creates :class:`Compressor`
    """

    __metaclass__ = ABCMeta

    def __init__(self, n_components):
        """
        Create a :class:`Compressor` instance

        Parameters
        ----------
        n_components : int > 0
            The number of components to keep
        """
        self.n_components = n_components

    @abstractmethod
    def __call__(self):
        """
        Return a new :class:`Compressor`.
        """
        pass


class Sampler(Compressor):
    """
    =======
    Sampler
    =======
    A :class:`Sampler` is a compressor which only takes a subset of the
    features

    Attributes
    ----------
    n_components : the number of features that the compressor produce
    """

    def __init__(self, n_components):
        Compressor.__init__(self, n_components)

    def fit(self, X, y):
        nbFeatures = X.shape[1]
        self._sublist = random.sample(xrange(nbFeatures),
                                      self.n_components)
        return self

    def transform(self, X):
        Xt = X.transpose()
        ls = []
        for index in self._sublist:
            ls.append(Xt[index])
        X2 = np.vstack(ls)
        return X2.transpose()


class SamplerFactory(CompressorFactory):
    """
    ==============
    SamplerFactory
    ==============
    Creates :class:`Sampler` compressor
    """

    def __init__(self, n_components):
        """
        Create a :class:`Compressor` instance

        Parameters
        ----------
        n_components : int > 0
            The number of components to keep
        """
        CompressorFactory.__init__(self, n_components)

    def __call__(self):
        return Sampler(self.n_components)


class PCAFactory(CompressorFactory):
    """
    ==========
    PCAFactory
    ==========
    Creates :class:`PCA` compressor
    """
    def __init__(self, n_components):
        """
        Create a :class:`Compressor` instance

        Parameters
        ----------
        n_components : int > 0
            The number of components to keep
        """
        CompressorFactory.__init__(self, n_components)

    def __call__(self):
#        return PCA(self.n_components)
        pass
