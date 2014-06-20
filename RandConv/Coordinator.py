# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
"""
:class:`Coordinator` are responsible for applying a feature extraction
mechanism to all the data contained in a imageBuffer and keeping the
consistency if it creates several feature vectors for one image
"""
from abc import ABCMeta, abstractmethod
import numpy as np

from Logger import Progressable
from TaskManager import SerialExecutor, ParallelExecutor
from Rescaler import Rescaler, MaxoutRescaler
from NumpyToPILConvertor import NumpyPILConvertor

__all__ = ["PixitCoordinator", "RandConvCoordinator",
           "CompressRandConvCoordinator", "LoadCoordinator"]


class Coordinator(Progressable):
    """
    ===========
    Coordinator
    ===========

    :class:`Coordinator` are responsible for applying a feature extraction
    mechanism to all the data contained in a imageBuffer and keeping the
    consistency if it creates several feature vectors for one image.

    The extraction mechanism is class dependent. It is the class
    responsability to document its policy.

    """

    __metaclass__ = ABCMeta

    def __init__(self, logger=None, verbosity=None, dtype=np.float32,
                 labeltype=np.uint8):
        Progressable.__init__(self, logger, verbosity)
        self._exec = SerialExecutor(logger, verbosity)
        self._dtype = dtype
        self._labeltype = labeltype
        if dtype is np.float:
            self._rescaler = Rescaler()
        else:
            self._rescaler = MaxoutRescaler(dtype)

    def parallelize(self, nbJobs=-1, tempFolder=None):
        """
        Parallelize the coordinator

        Parameters
        ----------
        nbJobs : int {>0, -1} (default : -1)
            The parallelization factor. If "-1", the maximum factor is used
        tempFolder : filepath (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :lib:`joblib` library)
        """
        self._exec = ParallelExecutor(nbJobs, self.getLogger(),
                                      self.verbosity, tempFolder)

    def process(self, imageBuffer, learningPhase=True):
        """
        Extracts the feature vectors for the images contained in the
        :class:`ImageBuffer`

        Abstract method to overload.

        Parameters
        ----------
        imageBuffer : :class:`ImageBuffer`
            The data to process
        learningPhase : bool (default : True)
            Specifies whether it is the learning phase. For some
            :class:`Coordinator`, this is not important but it might be for
            the stateful ones

        Return
        ------
        X : a numpy 2D array
            the N x M feature matrix. Each of the N rows correspond to an
            object and each of the M columns correspond to a variable
        y : an iterable of int
            the N labels corresponding to the N objects of X

        Note
        ----
        The method might provide several feature vectors per original image.
        It ensures the consistency with the labels and is explicit about
        the mapping.

        Implementation
        --------------
        The method :meth:`process` only "schedule" the work. The
        implementation of what is to be done is the responbility of the method
        :meth:`_onProcess`. It is this method that should be overloaded
        """
        self._nbColors = imageBuffer.nbBands()

        self.logMsg("Allocating the memory...", 35)

        nbFeatures = self.nbFeaturesPerObject(self._nbColors)
        nbObjs = self.nbObjects(imageBuffer)

        X = self._exec.createArray((nbObjs, nbFeatures), self._dtype)
        y = self._exec.createArray((nbObjs), self._labeltype)

        self.logMsg("X shape : "+str(X.shape), 35)
        self.logMsg("X dtype : "+str(X.dtype), 35)
        self.logSize("X total size : ", (X.size*X.itemsize), 35)

        self._exec.executeWithStart("Extracting features",
                                    self._onProcess, imageBuffer,
                                    learningPhase=learningPhase,
                                    XResult=X, yResult=y)

        return X, y

    @abstractmethod
    def _onProcess(self, imageBuffer, startIndex, learningPhase,
                   XResult, yResult):
        """
        Extracts the feature vectors for the images contained in the
        :class:`ImageBuffer`

        Abstract method to overload.

        Parameters
        ----------
        imageBuffer : :class:`ImageBuffer`
            The data to process
        learningPhase : bool (default : True)
            Specifies whether it is the learning phase. For some
            :class:`Coordinator`, this is not important but it might be for
            the stateful ones

        Return
        ------
        X : a numpy 2D array
            the N x M feature matrix. Each of the N rows correspond to an
            object and each of the M columns correspond to a variable
        y : an iterable of int
            the N labels corresponding to the N objects of X

        Note
        ----
        The method might provide several feature vectors per original image.
        It ensures the consistency with the labels and is explicit about
        the mapping.
        """
        pass

    def __call__(self, imageBuffer, learningPhase):
        """Delegate to :meth:`process`"""
        return self.process(imageBuffer, learningPhase)

    def clean(self, *args):
        self.setTask(1, "Cleaning up")
        for resource in args:
            self._exec.clean(resource)
        self.endTask()

    @abstractmethod
    def nbFeaturesPerObject(self, nbColors=1):
        """
        Return the number of features that this :class:`Coordinator` will
        produce per object
        """
        pass

    def nbObjects(self, imageBuffer):
        """
        Return the number of objects that this :class:`Coordinator` will
        produce
        """
        return len(imageBuffer)

    def getLogger(self):
        """
        Return
        ------
        logger : :class:`Logger`
            The internal logger (might be None)
        """
        return self._logger


class PixitCoordinator(Coordinator):
    """
    ================
    PixitCoordinator
    ================

    This coordinator uses a :class:`MultiSWExtractor` and a
    :class:`FeatureExtractor`. The first component extracts subwindows
    from the image while the second extract the features from each subwindow.

    Thus, this method creates several feature vectors per image. The number
    depends on the :class:`MultiSWExtractor` instance but are grouped
    contiguously.

    Note
    ----
    The :class:`FeatureExtractor` instance must be adequate wrt the image
    type
    """
    def __init__(self, multiSWExtractor, featureExtractor, logger=None,
                 verbosity=None):
        """
        Construct a :class:`PixitCoordinator`

        Parameters
        ----------
        multiSWExtractor : :class:`MultiSWExtractor`
            The component responsible for the extraction of subwindows
        featureExtractor: :class:`FeatureExtractor`
            The component responsible for extracting the features from
            each subwindow
        """
        Coordinator.__init__(self, logger, verbosity)
        self._multiSWExtractor = multiSWExtractor
        self._featureExtractor = featureExtractor

    def _onProcess(self, imageBuffer, startIndex, learningPhase,
                   XResult, yResult):
        """Overload"""

        convertor = NumpyPILConvertor()

        #Logging
        self.setTask(len(imageBuffer),
                     "PixitCoordinator loop for each image")
        #Init
        counter = 0
        index = startIndex * self.nbObjMultiplicator()

        #Main loop
        for image, label in imageBuffer:
            image = convertor.numpyToPIL(image)
            imgLs = self._multiSWExtractor.extract(image)
            #Filling the X and y
            for img in imgLs:
                tmpRes = self._featureExtractor.extract(
                    convertor.pILToNumpy(img))
                XResult[index] = self._rescaler(tmpRes)
                yResult[index] = label
                index += 1
            #Logging progress
            self.updateTaskProgress(counter)
            counter += 1

    def nbFeaturesPerObject(self, nbColors):
        height, width = self._multiSWExtractor.getFinalSize()
        return self._featureExtractor.nbFeaturesPerObject(height,
                                                          width,
                                                          nbColors)

    def nbObjMultiplicator(self):
        return self._multiSWExtractor.nbSubwidows()

    def nbObjects(self, imageBuffer):
        return len(imageBuffer)*self.nbObjMultiplicator()


class RandConvCoordinator(Coordinator):
    """
    ===================
    RandConvCoordinator
    ===================

    This coordinator uses a :class:`ConvolutionalExtractor` and a
    :class:`FeatureExtractor`. The first component extracts subwindows from
    the image applies filter to each subwindow and aggregate them while the
    second extract the features from each subwindow.

    Thus, this method creates several feature vectors per image. The number
    depends on the :class:`ConvolutionalExtractor` instance but are grouped
    contiguously.
    """

    def __init__(self, convolutionalExtractor, featureExtractor,
                 logger=None, verbosity=None):
        """
        Construct a :class:`RandConvCoordinator`

        Parameters
        ----------
        convolutionalExtractor : :class:`ConvolutionalExtractor`
            The component responsible for the extraction, filtering and
            aggregation of subwindows
        featureExtractor: :class:`FeatureExtractor`
            The component responsible for extracting the features from
            each filtered and aggregated subwindow

        Note
        ----
        The :class:`FeatureExtractor` instance must be adequate wrt the image
        type
        """
        Coordinator.__init__(self, logger, verbosity)
        self._convolExtractor = convolutionalExtractor
        self._featureExtractor = featureExtractor

    def _onProcess(self, imageBuffer, startIndex, learningPhase,
                   XResult, yResult):
        """Overload"""
        #Logging
        self.setTask(len(imageBuffer),
                     "RandConvCoordinator loop for each image")

        #Init
        counter = 0
        row = startIndex * self.nbObjMultiplicator()

        #Main loop
        for image, label in imageBuffer:
            #Get the subwindows x filters
            allSubWindows = self._convolExtractor.extract(image)

            #Accessing each subwindow set separately
            for filteredList in allSubWindows:
                #Accessing each filter separately for a given subwindow
                column = 0
                for filtered in filteredList:

                    #Extracting the features for each filter
                    filter_feature = self._featureExtractor.extract(filtered)
                    XResult[row, column:(column+len(filter_feature))] = filter_feature
                    column += len(filter_feature)

                #Corresponding label
                yResult[row] = label
                row += 1
            #Logging progress
            self.updateTaskProgress(counter)
            counter += 1

    def getFilters(self):
        """
        Return the filters used to process the image

        Return
        ------
        filters : iterable of numpy arrays
            The filters used to process the image, with the exclusion
            of the identity filter if the raw image was included
        """
        return self._convolExtractor.getFilters()

    def isImageIncluded(self):
        """
        Whether the raw image was included

        Return
        ------
        isIncluded : boolean
            True if the raw image was included
        """
        return self._convolExtractor.isImageIncluded()

    def _groupsInfo(self, nbFeatures):
        """
        Return information about the grouping of features (original image
        included if necessary)

        Parameters
        ----------
         nbFeatures : int > 0
            The number of features

        Return
        ------
        tuple = (nbFeatures, nbGroups, nbFeaturePerGroup)
        nbFeatures : int
            The number of features
        nbGroups : int
            The number of groups
        nbFeaturePerGroup : int
            The number of features per group
        """
        nbGroups = len(self.getFilters())*len(self._convolExtractor.getPoolers())
        if self.isImageIncluded():
            nbGroups += len(self._convolExtractor.getPoolers())
        nbFeaturePerGroup = nbFeatures // nbGroups
        return nbFeatures, nbGroups, nbFeaturePerGroup

    def featureGroups(self, nbFeatures):
        """
        Returns an iterable of start indices of the feature groups of X and
        the number of features

        Parameters
        ----------
        nbFeatures : int > 0
            The number of features

        Return
        ------
        tuple = (nbFeatures, nbGroups, ls)
        nbFeatures : int
            The number of features
        nbGroups : int
            The number of groups
        ls : iterable of int
            Returns an iterable of start indices of the feature groups of X
            and the number of features
        """
        nbFeatures, nbGroups, nbFeaturePerGroup = self._groupsInfo(nbFeatures)
        return (nbFeatures, nbGroups, xrange(0, nbFeatures+1,
                                             nbFeaturePerGroup))

    def importancePerFeatureGrp(self, classifier):
        """
        Computes the importance of each filter.

        Parameters
        ----------
        classifier : sklearn.ensemble classifier with
        :attr:`feature_importances_`
            The classifier (just) used to fit the model
        X : 2D numpy array
            The feature array. It must have been learnt by this
            :class:`ConvolutionalExtractor` with the given classifier
        Return
        ------
        pair = (importance, indices)
        importance : iterable of real
            the importance of each group of feature
        indices : iterable of int
            the sorted indices of importance in decreasing order
        """

        importance = classifier.feature_importances_
        nbFeatures, nbGroups, starts = self.featureGroups(len(importance))
        impPerGroup = []
        for i in xrange(nbGroups):
            impPerGroup.append(sum(importance[starts[i]:starts[i+1]]))

        return impPerGroup, np.argsort(impPerGroup)[::-1]

    def nbFeaturesPerObject(self, nbColors):
        # Number of filters * poolers
        nbGroups = len(self.getFilters())*len(self._convolExtractor.getPoolers())
        if self.isImageIncluded():
            nbGroups += len(self._convolExtractor.getPoolers())

        #Size of the extracted subwindows
        height, width = self._convolExtractor.getFinalSizePerSubwindow()

        #Number of features per individual subwindows
        nbFperGroup = self._featureExtractor.nbFeaturesPerObject(height,
                                                                 width,
                                                                 nbColors)
        #Total number of features
        return nbGroups*nbFperGroup

    def nbObjMultiplicator(self):
        return self._convolExtractor.getNbSubwindows()

    def nbObjects(self, imageBuffer):
        return len(imageBuffer)*self.nbObjMultiplicator()


class LoadCoordinator(RandConvCoordinator):
    """
    ===============
    LoadCoordinator
    ===============
    Pseudo :class:`Coordinator` which load a given file instead of processing
    the :class:`ImageBuffer` instance.
    """

    def __init__(self, rcCoordinator, learningFile, testingFile):
        """
        Construct a :class:`LoadCoordinator`

        Parameters
        ----------
        filename : file or str
            The path to the numpy file holding the 2D feature array
        """
        RandConvCoordinator.__init__(self, rcCoordinator._convolExtractor,
                                     rcCoordinator._featureExtractor,
                                     rcCoordinator._logger)
        self._learningFile = learningFile
        self._testingFile = testingFile
        self._exec= rcCoordinator._exec

    def process(self, imageBuffer=None, learningPhase=True):
        if learningPhase:
            X = np.load(self._learningFile)
        else:
            X = np.load(self._testingFile)
        labels = imageBuffer.getLabels()
        nbFactor = len(X) // len(labels)
        y = []
        for label in labels:
            y += [label]*nbFactor

        return X, y


class CompressRandConvCoordinator(RandConvCoordinator):
    """
    ===========================
    CompressRandConvCoordinator
    ===========================
    This :class:`RandConvCoordinator` class adds the feature of compressing
    the extracted features.
    The compression operates on each filter ouput separately and depends
    on the given :class:`Compressor`.

    It is possible not to include the first image in the compression.

    This is a stateful :class:`Coordinator`.
    """
    def __init__(self, convolutionalExtractor, featureExtractor,
                 compressorFactory, compressOriginalImage=True,
                 logger=None, verbosity=None):
        """
        Construct a :class:`CompressRandConvCoordinator`

        Parameters
        ----------
        convolutionalExtractor : :class:`ConvolutionalExtractor`
            The component responsible for the extraction, filtering and
            aggregation of subwindows
        featureExtractor: :class:`FeatureExtractor`
            The component responsible for extracting the features from
            each filtered and aggregated subwindow
        compressorFactory : callable(void)
            A factory method to create :class:`Compressor`. It must be
            parameterless

        Note
        ----
        The :class:`FeatureExtractor` instance must be adequate wrt the image
        type
        """
        RandConvCoordinator.__init__(self, convolutionalExtractor,
                                     featureExtractor, logger, verbosity)
        self._compressorFactory = compressorFactory
        self._compressImage = compressOriginalImage

    def process(self, imageBuffer, learningPhase):
        imgIncluded = self._convolExtractor.isImageIncluded()
        #Basic extraction
        X, y = RandConvCoordinator.process(self, imageBuffer)
        #Compression
        data = self._slice(X)
        if learningPhase:
            ls = self._exec("Learning and applying feature compression",
                            self._multiFitTransfrom, data, y)
            #Partial putting back together
            self._compressors = []
            Xs = []
            for lsi in ls:
                for x, c in lsi:
                    self._compressors.append(c)
                    Xs.append(x)
        else:
            pairs = zip(data, self._compressors)
            ls = self._exec("Feature compression", self._multiTransform, pairs)
            #Partial putting back together
            Xs = []
            for lsi in ls:
                for xi in lsi:
                    Xs.append(xi)
        #Put back the data together appropriately
        X2 = np.hstack(Xs)
        if imgIncluded and not self._compressImage:
            _, _, endImage = self._groupsInfo(X.shape[1])
            imgX = X.transpose()
            imgX = imgX[0:endImage]
            X2 = np.hstack((imgX.transpose(), X2))
        return X2, y

    def _doProcess(self, imageBuffer):
        pass

    def _multiFitTransfrom(self, Xs, y):
        """
        Learn and apply the compression on each element of Xs

        Parameters
        ----------
        Xs : iterable of feature arrays
            A list of features array. A given row of each feature vector
            is linked with the corresponding label, thus each element of Xs
            must have the same height (first index) which must also equal the
            length of y
        y : iterable of int
            the labels of each row

        Return
        ------
        ls : list of pairs = (Xi, comp_i)
        Xi : feature array
            The compressed feature array corresponding to the ith element
            of Xs
        comp_i : :class:`Compressor`
            The compressor which compressed the ith element of Xs
        """
        ls = []
        for X in Xs:
            compressor = self._compressorFactory()
            X2 = compressor.fit_transform(X, y)
            ls.append((X2, compressor))
        return ls

    def _multiTransform(self, ls):
        """
        Apply the compression previously learnt on each element of Xs

        Paramaters
        ----------
        ls : iterable of pairs = (Xi, comp_i)
        Xi : feature array
            A feature array to compress
        comp_i : :class:`Compressor`
            A :class:`Compressor` instance to use to compress the corresponding
            Xi

        Return
        ------
        Xs : iterable of feature arrays
            The compressed feature arrays (in the same order as given)
        """
        Xs = []
        for Xi, comp_i in ls:
            Xs.append(comp_i.transform(Xi))
        return Xs

    def _slice(self, X):
        """
        Slice the feature array appropriately for the compression

        Parameters
        ----------
        X : 2D numpy array
            The feature array
        Return
        ------
        A iterable subarray of the feature array as group of features generated
        by the same filter
        """
        XTranspose = np.array(X).transpose()
        slices = []
        nbFeatures, nbGroups, nbFeaturePerGroup = self._groupsInfo(X.shape[1])
        imgIncluded = self._convolExtractor.isImageIncluded()
        for i in xrange(nbGroups):
            Xtmp = XTranspose[i*nbFeaturePerGroup:(i+1)*nbFeaturePerGroup]
            slices.append(Xtmp.transpose())
        if imgIncluded and not self._compressImage:
            slices = slices[1:]
            self._imgSize = nbFeaturePerGroup
        return slices

    def featureGroups(self, nbFeatures):
        """
        Returns an iterabtransposele of start indices of the feature groups of X and
        the number of features

        Parameters
        ----------
        nbFeatures : int >0
            The number of features

        Return
        ------
        tuple = (nbFeatures, nbGroups, ls)
        nbFeatures : int
            The number of features
        nbGroups : int
            The number of groups
        ls : sequence of int
            Returns an iterable of start indices of the feature groups of X
            and the number of features
        """
        if (not self._compressImage) and self._convolExtractor.isImageIncluded():
            nbGroups = len(self.getFilters())  # Discouting the image
            nbFeaturePerGroup = (nbFeatures - self._imgSize) // nbGroups
            ls = [0]
            starts = range(self._imgSize, nbFeatures+1, nbFeaturePerGroup)
            return (nbFeatures, nbGroups+1, ls+starts)
        else:
            return RandConvCoordinator.featureGroups(self, nbFeatures)

#TODO XXX overload _grpInfo

if __name__ == "__main__":
    def _slice(X, nbFilter, imgIncluded, imgCompressed):
        XTranspose = X.transpose()
        slices = []
        nbGroups = nbFilter
        nbFeatures = X.shape[1]
        nbFeatPerGrp = nbFeatures / nbGroups
        for i in xrange(nbGroups):
            Xtmp = XTranspose[i*nbFeatPerGrp:(i+1)*nbFeatPerGrp]
            slices.append(Xtmp.transpose())
        if imgIncluded and not imgCompressed:
            slices = slices[1:]
        return slices

    X = np.arange(30).reshape(5, 6)
    print X
    ls = _slice(X, 3, False, False)
    for x in ls:
        print x
