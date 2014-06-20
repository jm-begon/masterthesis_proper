# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:38:37 2014

@author: Jm
"""
from abc import ABCMeta, abstractmethod
import numpy as np

class ImageFeatureExtractor:
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def extract(self, image):
        """Return a single feature vector"""
        pass
    
    def transform(self, images):
        newRow = []
        for image in images:		
            newRow.append(self.extract(image))
        return np.array(newRow)
    
    def __call__(self,image):
        return self.transform(image)     
        
class ImageMultiFeatureExtractor:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def extract(self, image):
        """Return a list of several feature vectors"""
        pass
    
    def transform(self, images):
        newRow = []
        for image in images:		
            newRow.append(self.extract(image))
        return newRow
    
    def __call__(self,image):
        return self.extract(image)  
        
        
class ImageLinearizationExtractor(ImageFeatureExtractor):
    def _lin(self, band):
        width, height = band.shape[0], band.shape[1]
        return band.reshape(width*height)

        
    def extract(self, img):
        if len(img.shape) == 2 :
            #Grey level img
            return self._lin(img)
        else :
            #RGB
            lin = []
            for depth in range(img.shape[2]):
                lin.append(self._lin(img[:,:,depth]))
            return np.hstack(lin)        