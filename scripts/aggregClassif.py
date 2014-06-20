# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : May 10 2014
"""
A script to aggregate several npz classification probability matrix
"""
import sys
import numpy as np

from sklearn.metrics import confusion_matrix

from CifarLoader import CifarFromNumpies
from ImageBuffer import FileImageBuffer, NumpyImageLoader

testingSetDir = "test/"
testingIndexFile = "0index"


if __name__ == "__main__":
    Y = np.load(sys.argv[0])
    for filename in sys.argv[1:]:
        Y += np.load(filename)

    y_pred = np.argmax(Y, axis=1)

    tsSize = len(y_pred)
    loader = CifarFromNumpies(testingSetDir, testingIndexFile)
    testingSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    testingSet = testingSet[0:tsSize]

    y_truth = testingSet.getLabels()

    accuracy = sum(map((lambda x, y: x == y), y_pred, y_truth))/float(len(y_truth))
    confMat = confusion_matrix(y_truth, y_pred)

    print "==================Classif aggregation================"
    print "Files :\n", sys.argv
    print "Accuracy", accuracy
    print "Confusion matrix :\n", confMat
