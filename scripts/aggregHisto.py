# -*- coding: utf-8 -*-
"""
Created on Sat May 10 12:25:39 2014

@author: Jm
"""
import sys
import time

import numpy as np
import scipy.sparse as sps

from sklearn.svm import LinearSVC

from Classifier import UnsupervisedVisualBagClassifier as uClassifier
from CifarLoader import CifarFromNumpies
from ImageBuffer import FileImageBuffer, NumpyImageLoader

#-----BagOfWords params + some SVC params
randomClassif = True
nbJobsEstimator = -1
verbose = 8
#=====DATA=====#
maxTestingSize = 10000

learningSetDir = "learn/"
learningIndexFile = "0index"

testingUse = 10000
testingSetDir = "test/"
testingIndexFile = "0index"


def formatBigNumber(num):
    revnum = str(num)[::-1]
    right = revnum
    rtn = ""
    for p in range((len(revnum)-1)//3):
        rtn += right[:3]+","
        right = right[3:]
    rtn += right
    return rtn[::-1]


def build_csr(filename):
    arrays = np.load(filename)
    data = arrays["data"]
    indices = arrays["indices"]
    indptr = arrays["indptr"]
    shape = arrays["shape"]

    return sps.csr_matrix((data, indices, indptr), shape=shape)

if __name__ == "__main__":
    ls = []
    for filename in sys.argv:
        ls.append(build_csr(filename))

    hist = sps.hstack(ls)
    histSize = hist.shape[0]

    randomState = None
    if not randomClassif:
        randomState = 100

    tsSize = testingUse
    if testingUse > maxTestingSize:
        tsSize = maxTestingSize

    #--SVM--
    baseClassif = LinearSVC(verbose=verbose, random_state=randomState)

    #--Classifier
    classifier = uClassifier(coordinator=None,
                             base_classifier=baseClassif,
                             n_jobs=nbJobsEstimator,
                             random_state=randomState,
                             verbose=verbose)

    #--Data--
    loader = CifarFromNumpies(learningSetDir, learningIndexFile)
    learningSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    learningSet = learningSet[0:histSize]

    loader = CifarFromNumpies(testingSetDir, testingIndexFile)
    testingSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    testingSet = testingSet[0:tsSize]

    #--Learning--#
    print "Starting learning"
    fitStart = time()
    y = learningSet.getLabels()
    classifier.fit_histogram(hist, y)
    fitEnd = time()
    print "Learning done", (fitEnd-fitStart), "seconds"
    sys.stdout.flush()

    #--Testing--#
    y_truth = testingSet.getLabels()
    predStart = time()
    y_pred = classifier.predict(testingSet)
    predEnd = time()
    accuracy = classifier.accuracy(y_pred, y_truth)
    confMat = classifier.confusionMatrix(y_pred, y_truth)

    print "==================Bag of Visual Words======================="
    print "--------Bag of words params + SVC----------"
    print "nbJobsEstimator", nbJobsEstimator
    print "verbose", verbose
    print "randomState", randomState
    print "------------Data---------------"
    print "LearningSet size", len(learningSet)
    print "TestingSet size", len(testingSet)
    print "-------------------------------"
    print "Fit time", (fitEnd-fitStart), "seconds"
    print "Classifcation time", (predEnd-predStart), "seconds"
    print "Accuracy", accuracy
    print "Leafs", formatBigNumber(classifier.histoSize)

    print "Confusion Matrix\n", confMat
