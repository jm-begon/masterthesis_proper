# -*- coding: utf-8 -*-
"""
Created on Sun May 04 13:02:09 2014

@author: Jm Begon
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    include_dirs = [np.get_include()],
#    ext_modules = cythonize("computeHistogram.pyx")
    ext_modules = cythonize("FastPooling.pyx")
)

