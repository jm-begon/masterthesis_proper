# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 10 2014
"""
A script to run the pixit classification.
"""
from time import time

from CoordinatorFactory import coordinatorPixitFactory
from sklearn.ensemble import ExtraTreesClassifier
from Classifier import Classifier
from CifarLoader import CifarFromNumpies
from ImageBuffer import FileImageBuffer, NumpyImageLoader


if __name__ == "__main__":

    #======HYPER PARAMETERS======#
    #PixitCoordinator param
    nbSubwindows = 10
    subwindowMinSizeRatio = 0.75
    subwindowMaxSizeRatio = 1.
    subwindowTargetWidth = 16
    subwindowTargetHeight = 16
    fixedSize = False
    nbJobs = 1
    verbosity = 10
    tempFolder = "temp/"

    #Extratree param
    nbTrees = 10
    maxFeatures = "auto"
    maxDepth = None
    minSamplesSplit = 2
    minSamplesLeaf = 1
    bootstrap = False
    nbJobsEstimator = 1
    randomState = None
    verbose = 10

    #=====DATA=====#
    maxLearningSize = 50000
    learningUse = 500
    learningSetDir = "learn/"
    learningIndexFile = "0index"
    maxTestingSize = 10000
    testingUse = 500
    testingSetDir = "test/"
    testingIndexFile = "0index"

    #======INSTANTIATING========#
    #--Pixit--
    pixitCoord = coordinatorPixitFactory(nbSubwindows,
                                         subwindowMinSizeRatio,
                                         subwindowMaxSizeRatio,
                                         subwindowTargetWidth,
                                         subwindowTargetHeight,
                                         fixedSize=fixedSize,
                                         nbJobs=nbJobs,
                                         verbosity=verbosity,
                                         tempFolder=tempFolder)

    #--Extra-tree--
    baseClassif = ExtraTreesClassifier(nbTrees,
                                       max_features=maxFeatures,
                                       max_depth=maxDepth,
                                       min_samples_split=minSamplesSplit,
                                       min_samples_leaf=minSamplesLeaf,
                                       bootstrap=bootstrap,
                                       n_jobs=nbJobsEstimator,
                                       random_state=randomState,
                                       verbose=verbose)

    #--Classifier
    classifier = Classifier(pixitCoord, baseClassif)

    #--Data--

    loader = CifarFromNumpies(learningSetDir, learningIndexFile)
    learningSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    learningSet = learningSet[0:learningUse]

    loader = CifarFromNumpies(testingSetDir, testingIndexFile)
    testingSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    testingSet = testingSet[0:testingUse]

    #=====COMPUTATION=====#
    #--Learning--#
    fitStart = time()
    classifier.fit(learningSet)
    fitEnd = time()

    #--Testing--#
    y_truth = testingSet.getLabels()
    predStart = time()
    y_pred = classifier.predict(testingSet)
    predEnd = time()
    accuracy = classifier.accuracy(y_pred, y_truth)
    confMat = classifier.confusionMatrix(y_pred, y_truth)

    print "========================================="
    print "--------SW extractor----------"
    print "#Subwindows", nbSubwindows
    print "subwindowMinSizeRatio", subwindowMinSizeRatio
    print "subwindowMaxSizeRatio", subwindowMaxSizeRatio
    print "subwindowTargetWidth", subwindowTargetWidth
    print "subwindowTargetHeight", subwindowTargetHeight
    print "fixedSize", fixedSize
    print "nbJobs", nbJobs
    print "--------ExtraTrees----------"
    print "nbTrees", nbTrees
    print "maxFeatures", maxFeatures
    print "maxDepth", maxDepth
    print "minSamplesSplit", minSamplesSplit
    print "minSamplesLeaf", minSamplesLeaf
    print "bootstrap", bootstrap
    print "nbJobsEstimator", nbJobsEstimator
    print "randomState", randomState
    print "------------Data---------------"
    print "LearningSet size", len(learningSet)
    print "TestingSet size", len(testingSet)
    print "-------------------------------"
    print "Fit time", (fitEnd-fitStart), "seconds"
    print "Fit time", (predEnd-predStart), "seconds"
    print "Accuracy", accuracy
    print "Confusion matrix", confMat
