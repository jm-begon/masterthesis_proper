# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 09 2014
"""
A small script to convert the cifar database to numpy files
"""
import cPickle
import numpy as np

from CifarLoader import CifarLoader
from ImageBuffer import ImageBuffer


if __name__ == "__main__":
    paths = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
             "data_batch_5"]
    destDirectory = "learn/"
#    paths = ["test_batch"]
#    destDirectory = "test/"


    labelFile = "0index"

    namesAndLabels = []

    for path in paths:

        cifar = CifarLoader(path, outputFormat=ImageBuffer.NUMPY_FORMAT)
        #cifar = cifar[:10]
        for i in xrange(len(cifar)):
            _, label, filename = cifar.getData(i)

            filename = ((filename.split("."))[0]) + ".npy"

            namesAndLabels.append((filename, label))

            data = cifar.getImage(i)

            np.save(destDirectory+filename, data)

    with open((destDirectory+labelFile), "wb") as indexFile:
        cPickle.dump(namesAndLabels, indexFile, 2)
