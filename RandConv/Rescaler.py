# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 13:57:26 2014

@author: Jm
"""
import numpy as np


class Rescaler:

    def __init__(self):
        pass

    def rescale(self, array):
        return array

    def __call__(self, array):
        return self.rescale(array)


class MaxoutRescaler(Rescaler):

    def __init__(self, dtype=np.uint8):
        try:
            info = np.iinfo(dtype)
        except:
            info = np.finfo(dtype)
        self._min = info.min
        self._max = info.max

    def rescale(self, array):
        return array.clip(self._min, self._max)
