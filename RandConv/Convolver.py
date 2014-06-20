# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
"""
A :class:`Convolver` applies a filter to an numpy array. Different shapes
of numpy arrays are tackled by different convolvers
"""

from scipy import signal as sg
import numpy as np

__all__ = ["Convolver", "RGBConvolver"]


class Convolver:
    """
    =========
    Convolver
    =========
    The base :class:`Convolver` performs classical convolution between
    a 2D numpy array and a 2D filter.

    Note
    ----
    The filter must be 2D numpy array.
    """

    def __init__(self, mode="same", boundary="fill", fillvalue=0):
        """
        Construct a :class:`Convolver`

        Parameters
        ----------
        mode : str {"same", "valid", "full"} (default : "same")
            A string indicating the size of the output:
            - full
                The output is the full discrete linear convolution of the
                inputs.
            - valid
                The output consists only of those elements that do not rely
                on the zero-padding.
            - same
                The output is the same size as in1, centered with respect to
                the ‘full’ output.
        boundary : str {"fill", "wrap", "symm"} (default : "fill")
            A flag indicating how to handle boundaries:
            - fill
                pad input arrays with fillvalue.
            - wrap
                circular boundary conditions.
            - symm
                symmetrical boundary conditions.
        fillvalue : scalar (default : 0)
            Value to fill pad input arrays with.
        """
        self._mode = mode
        self._boundary = boundary
        self._fillvalue = fillvalue

    def convolve(self, npImage, npFilter):
        """
        Return the 2D convolution of the image by the filter

        Parameters
        ----------
        npImage : 2D array like structure (usually numpy array)
            The image to filter
        npFilter : 2D array like structure (usually numpy array)
            The filter to apply by convolution

        Return
        ------
        filtered : 2D array
            The result of the convolution
        """
        return sg.convolve2d(npImage, npFilter, self._mode, self._boundary,
                             self._fillvalue)

    def __call__(self, npImage, npFilter):
        """
        Delegates to :meth:`convolve`
        """
        return self.convolve(npImage, npFilter)


class RGBConvolver(Convolver):
    """
    ============
    RGBConvolver
    ============
    The :class:`RGBConvolver` treats each colorband separately by performing
    classical convolution between each colorband and its respective filter.
    A colorband is suppose to be a 2D numpy array as are also the 2D filters.
    """

    def convolve(self, npImage, filters):
        """
        Return the 2D convolution of each colorband by its respective filter.

        Parameters
        ----------
        npImage : 3D numpy array where the dimensions represent respectively
        the height, the width and the colorbands. There must be 3 colorbands.
            The image to filter
        filters : a triplet of filters of the same size. The filters are
        2D array like structure (usually numpy array).
            The respective filters. They are applied in the same order to the
            colorbands.

        Return
        ------
        filtered :3D numpy array where the dimensions represent respectively
        the height, the width and the colorbands
            The result of the convolution of each colorband by its respective
            filter
        """
        redFilter, greenFilter, blueFilter = filters
        red, green, blue = npImage[:,:,0], npImage[:,:,1], npImage[:,:,2]

        newRed = sg.convolve2d(red, redFilter, "same")
        newGreen = sg.convolve2d(green, greenFilter, "same")
        newBlue = sg.convolve2d(blue, blueFilter, "same")

        return np.dstack((newRed, newGreen, newBlue))

if __name__ == "__main__":
    test=True
    if test:
        imgpath = "lena.png"
        try:
            import Image
        except:
            from PIL import Image
        img = np.array(Image.open(imgpath))
        red = img[:,:,0]
        
        gx = np.array([[-1, 0, 1], [-2,0,2], [-1,0,1]])
        
        redFiltered = Convolver().convolve(red, gx)
        
        imgFiltered = RGBConvolver().convolve(img, (gx,gx,gx))
        