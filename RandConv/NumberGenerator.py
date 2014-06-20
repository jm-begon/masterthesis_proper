# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 22 2014
"""
A set of number generator
"""
from sklearn.utils import check_random_state
from scipy.stats import norm

__all__ = ["NumberGenerator", "IntegerUniformGenerator", "OddUniformGenerator",
           "GaussianNumberGenerator", "ClippedGaussianRNG",
           "CustomDiscreteNumberGenerator", "ConstantGenerator"]


class NumberGenerator:
    """
    ===============
    NumberGenerator
    ===============
    A random number generator which draws number between two bounds :
    [min, max)
    This base class returns uniform real number
    """
    def __init__(self, minVal=0, maxVal=1, seed=None):
        """
        Construct a :class:`NumberGenerator`

        Parameters
        ----------
        minVal : float (default : 0)
            The minimum value from which to draw
        maxVal : float (default : 1)
            The maximum value from which to draw
        seed : int or None (default : None)
            if seed is int : initiate the random generator with this seed
        """
        self._randGen = check_random_state(seed)
        self._min = minVal
        self._max = maxVal

    def _doGetNumber(self, minVal, maxVal):
#        """
#        Return a random number comprise between [minVal, maxVal)
#
#        Parameters
#        ----------
#        minVal : float
#            The minimum value from which to draw
#        maxVal : float
#            The maximum value from which to draw
#
#        Return
#        ------
#        rand : float
#            A random number comprise between [minVal, maxVal)
#        """
        return minVal+self._randGen.rand()*(maxVal-minVal)

    def getNumber(self, minVal=None,  maxVal=None):
        """
        Return a random number comprise between [minVal, maxVal)

        Parameters
        ----------
        minVal : float/None (default : None)
            The minimum value from which to draw
            if None : use the instance minimum
        maxVal : float/None (default : None)
            The maximum value from which to draw
            if None : use the instance maximum

        Return
        ------
        rand : float
            A random number comprise between [minVal, maxVal)
        """
        if minVal is None:
            minVal = self._min
        if maxVal is None:
            maxVal = self._max
        return self._doGetNumber(minVal, maxVal)


class GaussianNumberGenerator(NumberGenerator):

    """
    =======================
    GaussianNumberGenerator
    =======================
    Generate real number from a Gaussian law
    """

    def __init__(self, minVal=0, maxVal=1, seed=None, outsideRange=0.05):
        """
        Creates a :class:`GaussianNumberGenerator``instance.
        The mean is (minVal + maxVal)/2.
        The stdev is computed thanks to the outsideRange

        Parameters
        ----------
        minVal : float (default : 0)
            The minimum value from which to draw
        maxVal : float (default : 1)
            The maximum value from which to draw
        seed : int or None (default : None)
            if seed is int : initiate the random generator with this seed
        outsideRange : float [0, 1)
            The probability of generating a value which is outside of the range
            [minVal, maxVal)
        """
        NumberGenerator.__init__(self, minVal, maxVal, seed)
        inRange = 1-(outsideRange/2.)
        self._k = norm.ppf(inRange)

    def _doGetNumber(self, minVal, maxVal):
        mean = (maxVal + minVal)/2.
        std = (mean - minVal)/self._k
        val = self._randGen.normal(mean, std)
        return val


class ClippedGaussianRNG(GaussianNumberGenerator):
    """
    =======================
    GaussianNumberGenerator
    =======================
    Generate real number from a Gaussian law, clipped at the given range
    """
    def __init__(self, minVal=0, maxVal=1, seed=None, outsideRange=0.05):
        GaussianNumberGenerator.__init__(self, minVal, maxVal, seed,
                                         outsideRange)

    def _doGetNumber(self, minVal, maxVal):
        val = GaussianNumberGenerator._doGetNumber(self, minVal, maxVal)
        if val < minVal:
            return minVal
        if val > maxVal:
            return maxVal
        return val


class CustomDiscreteNumberGenerator(NumberGenerator):
    """
    =============================
    CustomDiscreteNumberGenerator
    =============================
    Generate a number of a predifine set whose elements are given a predifine
    probability.

    Note
    ----
    If the "probabilities" associated to the elements does not sum up, there
    are scaled to do so.
    """

    def __init__(self, lsOfPairs, seed=None):
        """
        Creates a :class:`GaussianNumberGenerator``instance

        Parameters
        ----------
        lsOfPairs : iterable of pairs (number, probability)
            number : number
                an element of the set from which to draw
            probability : float
                the probability of the element being chosen at each draw
        seed : int or None (default : None)
            if seed is int : initiate the random generator with this seed
        """
        self._vals = [v for v, p in lsOfPairs]
        probs = [p for v, p in lsOfPairs]
        sumP = sum(probs)
        probs = [p/sumP for p in probs]
        self._cumulProbs = [0]*len(probs)
        for i in xrange(len(probs)-1):
            self._cumulProbs[i+1] = self._cumulProbs[i] + probs[i]
        NumberGenerator.__init__(self, 0, 1, seed)

    def getNumber(self, minVal=0, maxVal=1):
        prob = self._randGen.rand()
        index = self._search(prob)
        return self._vals[index]

    def _search(self, prob):
        """
        Return the index of self._cumulProbs such that
        p \in [self._cumulProbs[index], self._cumulProbs[index+1])
        """
        #Linear search for now
        for i in xrange(len(self._cumulProbs)-1):
            if prob >= self._cumulProbs[i] and prob < self._cumulProbs[i+1]:
                return i
        #Should not happen
        if prob < self._cumulProbs[0]:
            return 0
        return len(self._cumulProbs)-1


class IntegerUniformGenerator(NumberGenerator):
    """
    =======================
    IntegerUniformGenerator
    =======================
    A random number generator which draws number between two bounds :
    [min, max)
    This class returns uniform integer number
    """
    def _doGetNumber(self, minVal, maxVal):
        return self._randGen.randint(minVal, maxVal)


class OddUniformGenerator(IntegerUniformGenerator):
    """
    =======================
    IntegerUniformGenerator
    =======================
    A random number generator which draws number between two bounds :
    [min, max)
    This class returns uniform odd integer number
    """
    def _doGetNumber(self, minVal, maxVal):
        if maxVal % 2 == 0:
            maxVal -= 1
        if minVal % 2 == 0:
            minVal += 1
        return minVal + 2*int(self._randGen.rand()*((maxVal - minVal)/2+1))


class ConstantGenerator(NumberGenerator):
    """
    =================
    ConstantGenerator
    =================
    A not so random number generator. "Generate" a constant number while
    preserving the :class:`NumberGenerator` interface.
    """
    def __init__(self, constant):
        self._const = constant

    def getNumber(self, minVal=None,  maxVal=None):
        return self._const
