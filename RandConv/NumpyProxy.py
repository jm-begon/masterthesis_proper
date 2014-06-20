# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 14:05:07 2014

@author: Jm
"""
import numpy as np
from functools import partial


def className(obj):
    return obj.__class__.__name__


def decorate(proxy, functionName, *args1, **kwargs1):
    print functionName
    if proxy._np is None:
        #Using the approrpiate function
        args = proxy._loadArgs
        kwargs = proxy._loadDict
        if len(args) == 0:
            if len(kwargs) == 0:
                proxy._np = proxy._loadFunc()
            else:
                proxy._np = proxy._loadFunc(**kwargs)
        elif len(kwargs) == 0:
            proxy._np = proxy._loadFunc(*args)
        else:
            proxy._np = proxy._loadFunc(*args, **kwargs)
        # TODO XXX undecorate
    return getattr(proxy._np, functionName)(*args1, **kwargs1)


class NumpyProxy(np.ndarray):

    def __new__(cls, shape, dtype=float, loadingFunction=(lambda x: x),
                loadingArgs=[], loadingDict={},  buffer=None, offset=0,
                strides=None, order=None, pxyClass=np.ndarray):

        obj = np.ndarray.__new__(cls, shape, dtype, buffer, offset,
                                 strides, order)

#        #Decoration
#        for name in dir(np.ndarray):
#            attr = getattr(np.ndarray, name)
#            if callable(attr):
#                if name is "__class__":
#                    continue
#                setattr(obj, name, partial(decorate, obj, name))

        #Loading mechanism
        obj._loadFunc = loadingFunction
        obj._loadArgs = loadingArgs
        obj._loadDict = loadingDict
        obj._np = None
        return obj



#    def __array_finalize__(self, obj):
#        if obj is None :
#            return

    def __decorate(self, functionName):
        print functionName
        if self._np is None:
            self._np = self._loadFunc(*self._loadArgs, **self._loadDict)
            # TODO XXX undecorate
        return getattr(self._np, functionName)
