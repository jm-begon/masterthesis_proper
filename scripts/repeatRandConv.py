# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
"""
Apply randConvRepeatedly
"""

import numpy as np
import randConvCifar as rc


def variability(importance):
    return np.std(importance)


if __name__ == "__main__":
    nbRun = 50
    acc = []
    importances = []
    for i in xrange(nbRun):
        accTmp, _, imp, _ = rc.run()
        acc.append(accTmp)
        importances.append(variability(imp))

    print "Accuracies"
    print acc
    npAcc = np.array(acc)
    print "Mean accuracy", npAcc.mean()
    print "Stdev accuracy",  npAcc.std()
    print "---------------------------------"
    print "Feature importance std"
    print importances
