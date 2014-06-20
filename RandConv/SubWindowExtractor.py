# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
"""
A set of subwindow extractor
"""
try:
    import Image
except ImportError:
    from PIL import Image

__all__ = ["SubWindowExtractor", "FixTargetSWExtractor",
           "FixImgSWExtractor", "MultiSWExtractor"]


class SubWindowExtractor:
    """
    ==================
    SubWindowExtractor
    ==================
    A :class:`SubWindowExtractor` extract subwindows from an image and resize
    them to a given shape.

    The size and location of a given subwindow are drawn randomly

    Interpolation
    -------------
    The resizing step needs a interpolation algorithm :
    INTERPOLATION_NEAREST
        Nearest neighbor interpolation
    INTERPOLATION_BILINEAR
        bilinear interpolation
    INTERPOLATION_CUBIC
        bicubic interpolation
    INTERPOLATION_ANTIALIAS
        antialisaing interpolation
    """
    INTERPOLATION_NEAREST = 1
    INTERPOLATION_BILINEAR = 2
    INTERPOLATION_CUBIC = 3
    INTERPOLATION_ANTIALIAS = 4

    def __init__(self, minSize, maxSize, targetWidth, targetHeight,
                 interpolation, numberGenerator):
        """
        Construct a :class:`SubWindowExtractor`

        Parameters
        ----------
        minSize : float 0 < minSize <= 1
            The minimum size of subwindow express as the size ratio with
            the original image
        maxSize : float minSize <= maxSize <= 1
            The maximum size of subwindow express as the size ratio with
            the original image
        targetWidth : int > 0
            The width of the subwindow after resizing
        targetHeight : int > 0
            the height of the subwindow after resizing
        interpolation : int {INTERPOLATION_NEAREST, INTERPOLATION_BILINEAR,
        INTERPOLATION_CUBIC, INTERPOLATION_ANTIALIAS}
            The reintorpolation mechanism
        numberGenerator : :class:`NumberGenerator`
            The random number generator used for drawing the subwindows. It
            draws the height and width of the subwindow (respecting the
            original ratio) and then draws the location. Real number generator
            are fine and will be casted into int
        """
        self._minSize = minSize
        self._maxSize = maxSize
        self._targetWidth = targetWidth
        self._targetHeight = targetHeight
        self._numGen = numberGenerator
        self.setInterpolation(interpolation)

    def setInterpolation(self, interpolation):
        """
        Set the interpolation algorithm for this :class:`SubWindowExtractor`
        instance.

        Paramters
        ---------
        interpolation : int {INTERPOLATION_NEAREST, INTERPOLATION_BILINEAR,
        INTERPOLATION_CUBIC, INTERPOLATION_ANTIALIAS}
            The reintorpolation mechanism
        """
        if interpolation == SubWindowExtractor.INTERPOLATION_NEAREST:
            pil_interpolation = Image.NEAREST
        elif interpolation == SubWindowExtractor.INTERPOLATION_BILINEAR:
            pil_interpolation = Image.BILINEAR
        elif interpolation == SubWindowExtractor.INTERPOLATION_CUBIC:
            pil_interpolation = Image.CUBIC
        elif interpolation == SubWindowExtractor.INTERPOLATION_ANTIALIAS:
            pil_interpolation = Image.ANTIALIAS
        else:
            pil_interpolation = Image.BILINEAR
        self._interpolation = pil_interpolation

    def getCropBox(self, width, height):
        """
        Draws a new crop box

        Paramters
        ---------
        width : int > 0
            the width of the image on which the cropbox will be used
        height: int > 0
            the height of the image on which the cropbox will be used

        Return
        ------
        tuple = (px, py, dx, dy)
        px : int
            The x-coordinate of the upper left pixel of the cropbox
        py : int
            The y-coordinate of the upper left pixel of the cropbox
        dx : int
            the x-coordinate of the lower right pixel of the cropbox
        dy : int
            the y-coordinate of the lower right pixel of the cropbox
        """
        if width < height:
            ratio = 1. * self._targetHeight / self._targetWidth
            min_width = self._minSize * width
            max_width = self._maxSize * width

            if min_width * ratio > height:
                raise ValueError

            if max_width * ratio > height:
                max_width = height / ratio

            cropWidth = self._numGen.getNumber(min_width, max_width)
            cropHeight = ratio * cropWidth

        else:
            ratio = 1. * self._targetWidth / self._targetHeight
            min_height = self._minSize * height
            max_height = self._maxSize * height

            if min_height * ratio > width:
                raise ValueError

            if max_height * ratio > width:
                max_height = width / ratio

            cropHeight = self._numGen.getNumber(min_height, max_height)
            cropWidth = ratio * cropHeight

        if cropWidth == 0:
            cropWidth = 1
        if cropHeight == 0:
            cropHeight = 1

        # Draw a random position
        px = int(self._numGen.getNumber(0, width-cropWidth))
        py = int(self._numGen.getNumber(0, height-cropHeight))

        # Crop subwindow
        return (px, py, int(px+cropWidth), int(py+cropHeight))

    def extractWithBoxes(self, image):
        """
        Extract a subwindow of an image

        Parameters
        ----------
        image : PIL.Image
            The image from which to extract the subwindow

        Return
        ------
        pair = (subwindow, box)
        subwindow : PIL.Image
            The subwindow extracted from the original image
        box = (px, py, dx, dy) the croping box
        px : int
            The x-coordinate of the upper left pixel of the cropbox
        py : int
            The y-coordinate of the upper left pixel of the cropbox
        dx : int
            the x-coordinate of the lower right pixel of the cropbox
        dy : int
            the y-coordinate of the lower right pixel of the cropbox
        """
        # Draw a random window
        width, height = image.size
        height = image.shape[0]
        width = image.shape[1]
        try:
            box = self.getCropBox(width, height)
        except CorpLargerError:
            #subwindow larger than image, so we simply resize original image
            #to target sizes
            subWindow = image.resize((self._targetWidth, self._targetHeight),
                                     self._interpolation)
            return subWindow, box

        return self.cropAndResize(image, box)

    def extract(self, image):
        """
        Extract a subwindow of an image

        Parameters
        ----------
        image : PIL.Image
            The image from which to extract the subwindow

        Return
        ------
        subwindow : PIL.Image
            The subwindow extracted from the original image
        """
        sw, box = self.extractWithBoxes(image)
        return sw

    def cropAndResize(self, image, cropbox):
        """
        Apply image cropping and resize image thanks to the instance
        reinterpolation mechanism

        Parameters
        ----------
        image : PIL.Image
            The image from which to extract the subwindow
        box = (px, py, dx, dy) the croping box
            px : int
                The x-coordinate of the upper left pixel of the cropbox
            py : int
                The y-coordinate of the upper left pixel of the cropbox
            dx : int
                the x-coordinate of the lower right pixel of the cropbox
            dy : int
                the y-coordinate of the lower right pixel of the cropbox

        Return
        ------
        pair = (sub_window, cropbox)
        sub_window : PIL.Image
            The resized cropped image
        cropbox : the box itself
        """
        sub_window = image.crop(cropbox).resize((self._targetWidth,
                                                self._targetHeight),
                                                self._interpolation)
        return sub_window, cropbox

    def getFinalSize(self):
        """
        Return the final size of the windows

        Return
        ------
        pair = (height, width)
            height : int > 0
                The height of the subwindows
            width : int > 0
                The width of the subwindows
        """
        return self._targetHeight, self._targetWidth


class FixTargetSWExtractor(SubWindowExtractor):
    """
    ====================
    FixTargetSWExtractor
    ====================
    This subwindow extractor does not draw the size of the subwindow but
    directly uses the target size.
    """
    def __init__(self, targetWidth, targetHeight, interpolation,
                 numberGenerator):
        """
        Construct a :class:`FixTargetSWExtractor` instance.

        Parameters
        ----------
        targetWidth : int > 0
            The width of the subwindow after resizing
        targetHeight : int > 0
            the height of the subwindow after resizing
        interpolation : int {INTERPOLATION_NEAREST, INTERPOLATION_BILINEAR,
        INTERPOLATION_CUBIC, INTERPOLATION_ANTIALIAS}
            The reintorpolation mechanism
        numberGenerator : :class:`NumberGenerator`
            The random number generator used for drawing the subwindow
            locations. Real number generator are fine and will be casted
            into int
        """
        self._targetHeight = targetHeight
        self._targetWidth = targetWidth
        self._numGen = numberGenerator
        self.setInterpolation(interpolation)

    def getCropBox(self, width, height):

        cropWidth = self._targetWidth
        cropHeight = self._targetHeight
        if cropWidth > width or cropHeight > height:
            raise CorpLargerError("Crop larger than image")

         # Draw a random position
        px = int(self._numGen.getNumber(0, width-cropWidth))
        py = int(self._numGen.getNumber(0, height-cropHeight))

        # Crop subwindow
        return (px, py, int(px + cropWidth), int(py + cropHeight))


class FixImgSWExtractor(SubWindowExtractor):
    """
    ====================
    FixImgSWExtractor
    ====================
    This subwindow extractor works with images of a given width and height
    """
    def __init__(self, imageWidth, imageHeight, minSize, maxSize,
                 targetWidth, targetHeight, interpolation,
                 numberGenerator):
        """
        Construct a :class:`FixImgSWExtractor`

        Parameters
        ----------
        imageWidth : int > 0
            The image width
        imageHeight : int > 0
            the image height
        minSize : float 0 < minSize <= 1
            The minimum size of subwindow express as the size ratio with
            the original image
        maxSize : float minSize <= maxSize <= 1
            The maximum size of subwindow express as the size ratio with
            the original image
        targetWidth : int > 0
            The width of the subwindow after resizing
        targetHeight : int > 0
            the height of the subwindow after resizing
        interpolation : int {INTERPOLATION_NEAREST, INTERPOLATION_BILINEAR,
        INTERPOLATION_CUBIC, INTERPOLATION_ANTIALIAS}
            The reintorpolation mechanism
        numberGenerator : :class:`NumberGenerator`
            The random number generator used for drawing the subwindows. It
            draws the height and width of the subwindow (respecting the
            original ratio) and then draws the location. Real number generator
            are fine and will be casted into int
        """
        self._imgWidth = imageWidth
        self._imgHeight = imageHeight
        self._minSize = minSize
        self._maxSize = maxSize
        self._targetWidth = targetWidth
        self._targetHeight = targetHeight
        self._numGen = numberGenerator
        self.setInterpolation(interpolation)
        self._computeMinMaxHeight(imageWidth, imageHeight)

    def _computeMinMaxHeight(self, width, height):
        ratio = 1. * self._targetWidth / self._targetHeight
        min_height = self._minSize * height
        max_height = self._maxSize * height

        if min_height * ratio > width:
            raise ValueError

        if max_height * ratio > width:
            max_height = width / ratio

        self._minHeight = min_height
        self._maxHeight = max_height
        self._ratio = ratio

    def getCropBox(self, width=None, height=None):

        cropHeight = self._numGen.getNumber(self._minHeight, self._maxHeight)
        cropWidth = self._ratio * cropHeight

        if cropWidth == 0:
            cropWidth = 1
        if cropHeight == 0:
            cropHeight = 1

        # Draw a random position
        px = int(self._numGen.getNumber(0, self._imgWidth-cropWidth))
        py = int(self._numGen.getNumber(0, self._imgHeight-cropHeight))

        # Crop subwindow
        return (px, py, int(px + cropWidth), int(py + cropHeight))


class CorpLargerError(Exception):
    """
    ===============
    CorpLargerError
    ===============
    An exception class which represents the fact that a cropping box is
    larger than the image to crop
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class MultiSWExtractor:
    """
    ================
    MultiSWExtractor
    ================
    A subwindow extractor which extracts severals subwindows per image.

    See :meth:`refresh`.
    """
    def __init__(self, subwindowExtractor, nbSubwindows, autoRefresh=False):
        """
        Construct a :class:`MultiSWExtractor`

        Parameters
        ----------
        subwindowExtractor : :class:`SubWindowExtractor`
            The instance which will extract each subwindow
        nbSubwindows : int > 0
            The number of subwindow to extract
        autoRefresh : boolean (default : False)
            if true, refreshes the set of cropboxes for each image
            See :meth:`refresh`.
        """
        self._swExtractor = subwindowExtractor
        self._nbSW = nbSubwindows
        self._autoRefresh = autoRefresh

    def __len__(self):
        return self._nbSW

    def nbSubwidows(self):
        """
        Return the number of subwindows that this instance will extract
        """
        return self._nbSW

    def refresh(self, width, height):
        """
        Change/refresh/update the set of cropboxes.

        Parameters
        ----------
        width : int > 0
            The image width
        height : int > 0
            the image height
        """
        boxes = []
        for i in xrange(self._nbSW):
            boxes.append(self._swExtractor.getCropBox(width, height))
        self._boxes = boxes

    def extractWithBoxes(self, image):
        """
        Extract a subwindow of an image

        Parameters
        ----------
        image : PIL.Image
            The image from which to extract the subwindows

        Return
        ------
        list : list of pairs = (subwindow, box)
        subwindow : PIL.Image
            The subwindow extracted from the original image
        box = (px, py, dx, dy) the croping box
        px : int
            The x-coordinate of the upper left pixel of the cropbox
        py : int
            The y-coordinate of the upper left pixel of the cropbox
        dx : int
            the x-coordinate of the lower right pixel of the cropbox
        dy : int
            the y-coordinate of the lower right pixel of the cropbox
        """
        #Testing auto refresh
        if self._autoRefresh:
            width, height = image.size
            self.refresh(width, height)
        #Extracting the boxes
        subwindowsAndBoxes = []
        for box in self._boxes:
            subwindowsAndBoxes.append(
                self._swExtractor.cropAndResize(image, box))
        return subwindowsAndBoxes

    def extract(self, image):
        """
        Extract a subwindow of an image

        Parameters
        ----------
        image : PIL.Image
            The image from which to extract the subwindows

        Return
        ------
        list of subwindows
        subwindow : PIL.Image
            The subwindow extracted from the original image
        """
        #Testing auto refresh
        if self._autoRefresh:
            width, height = image.size
            self.refresh(width, height)
        #Extracting the boxes
        subWindows = []
        for box in self._boxes:
            sw, _ = self._swExtractor.cropAndResize(image, box)
            subWindows.append(sw)
        return subWindows

    def getFinalSize(self):
        """
        Return the final size of the windows

        Return
        ------
        pair = (height, width)
            height : int > 0
                The height of the subwindows
            width : int > 0
                The width of the subwindows
        """
        return self._swExtractor.getFinalSize()


if __name__ == "__main__":

    test = True

    if test:
        imgpath = "lena.png"
        from NumberGenerator import NumberGenerator
        try:
            import Image
        except:
            from PIL import Image
        img = Image.open(imgpath)
        width, height = img.size
        swExt = SubWindowExtractor(0.5,1.,256,250, SubWindowExtractor.INTERPOLATION_BILINEAR, NumberGenerator())

        sw1,box1 = swExt.extract(img)

        mExt = MultiSWExtractor(swExt, 10)
        mExt.refresh(width, height)

        sws = mExt.extract(img)



        swExt = FixImgSWExtractor(width, height, 0.5,1.,256,250, SubWindowExtractor.INTERPOLATION_BILINEAR, NumberGenerator())

        sw2,box2 = swExt.extract(img)

        mExt = MultiSWExtractor(swExt, 10)
        mExt.refresh(width, height)

        sws2 = mExt.extract(img)
