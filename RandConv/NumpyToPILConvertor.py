# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 08 2014
"""
A convertor between the two image format we use : PIL and numpy
See :mod:`ImageBuffer` for more information
"""
try:
    import Image
except ImportError:
    from PIL import Image
import numpy as np

__all__ = ["NumpyPILConvertor", "NumpyToPILConvertor", "PILToNumpyConvertor"]


class NumpyPILConvertor:
    """
    ==================
    NumpyPILConvertor
    ==================
    A convertor between PIL image and numpy
    """

    def numpyToPIL(self, npImage):
        """
        Convert a numpy image into a PIL image
        Parameters
        ----------
        npImage : a numpy image
            The image to convert
        Return
        ------
        pilImage : a :class:`PIL.Image`
            The converted image
        """
        if isinstance(npImage, Image.Image):
            return npImage
        return Image.fromarray(np.uint8(npImage.clip(0, 255)))

    def pILToNumpy(self, pilImg):
        """
        Convert a a PIL image into numpy image
        Parameters
        ----------
        pilImage : a :class:`PIL.Image`
            The image to convert
        Return
        ------
        npImage : a numpy image
            The converted image
        """
        if isinstance(pilImg, np.ndarray):
            return pilImg
        return np.array(pilImg)


class NumpyToPILConvertor(NumpyPILConvertor):
    """
    ===================
    NumpyToPILConvertor
    ===================
    A numpy to PIL image convertor
    """
    def __call__(self, img):
        """Delegates to :meth:`numpyToPil`"""
        return self.numpyToPIL(img)


class PILToNumpyConvertor(NumpyPILConvertor):
    """
    ===================
    PILToNumpyConvertor
    ===================
    A PIL image to numpy convertor
    """
    def __call__(self, img):
        """Delegates to :meth:`pILToNumpy`"""
        return self.pILToNumpy(img)
