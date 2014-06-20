# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 08 2014
"""
Several tools for parallel computation
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import copy_reg
import types


from sklearn.externals.joblib import Parallel, delayed, cpu_count
from Logger import Progressable
from NumpyFactory import NumpyFactory

__all__ = ["TaskSplitter", "TaskExecutor", "SerialExecutor",
           "ParallelExecutor"]


def reduceMethod(m):
    """Adds the capacity to pickle method of objects"""
    return (getattr, (m.__self__, m.__func__.__name__))

copy_reg.pickle(types.MethodType, reduceMethod)


class TaskSplitter:
    """
    ===========
    TaskSplitter
    ===========
    A toolkit for preprocessing parallel computation
    """
    def computePartition(self, nbTasks, dataSize):
        """
        Compute data partitioning for parallel computation :
        min(nbTasks, dataSize)

        Parameters
        ----------
        nbTasks : int (!=0)
            If >0 : the parallelization factor.
            If <0 : nbTasks = #cpu+nbTasks+1 (-1 -> nbTasks = #cpu)
        dataSize : int > 0
            The size of the data to process

        Return
        ------
        triplet = (nbTasks, counts, starts)
        nbTasks : int
            The final parallelization factor. It is computed as
            min(#cpu/nbTasks, dataSize)
        counts : list of int
            The number of data pieces for each parallel task
        starts : list of int
            The start indexes of the data for each parallel task
        """
        if nbTasks < 0:
            cpu = cpu_count()+nbTasks+1
            if cpu <= 0:
                cpu = 1
            nbTasks = min(cpu, dataSize)
        else:
            if nbTasks == 0:
                nbTasks = 1
            nbTasks = min(nbTasks, dataSize)

        counts = [dataSize / nbTasks] * nbTasks

        for i in xrange(dataSize % nbTasks):
            counts[i] += 1

        starts = [0] * (nbTasks + 1)

        for i in xrange(1, nbTasks + 1):
            starts[i] = starts[i - 1] + counts[i - 1]

        return nbTasks, counts, starts

    def partition(self, nbTasks, data):
        """
        Partition the data for parallel computation.

        Parameters
        ----------
        nbTasks : int {-1, >0}
            The parallelization factor. If -1 : the greatest factor is chosen
        data : list
            The data to partition

        Return
        ------
        tuple = (nbTasks, dataParts, starts)
        nbTasks : int
            The final parallelization factor. It is computed as
            min(#cpu/nbTasks, dataSize)
        dataParts : list of slices
            each element of dataParts is a contiguous slice of data : the
            partition for a parallel computation unit
        starts : list of int
            The start indices corresponding to the dataParts :
            dataParts[i] = data[starts[i]:starts[i+1]
        """
        nbTasks, counts, starts = self.computePartition(nbTasks, len(data))
        dataParts = []
        for i in xrange(nbTasks):
            dataParts.append(data[starts[i]:starts[i + 1]])
        return nbTasks, dataParts, starts


class TaskExecutor(Progressable):
    """
    ===========
    TaskExecutor
    ===========
    A class responsible for carrying out submitted tasks
    """
    __metaclass__ = ABCMeta

    def __init__(self, logger=None, verbosity=10):
        """
        Creates a :class:`TaskExecutor`
        """
        Progressable.__init__(self, logger, verbosity)

    @abstractmethod
    def execute(self, descr, function, data, *args, **kwargs):
        """
        Get the result of the task directly

        Parameters
        ----------
        desc : str
            A string describing the task
        function : callable
            The function to process the task. The function must be able to
            work on any subset of the data
        data : an iterable of piece of data
            The data to process
        args : iterable
            Parameters to pass to the function
        kwargs: dictionnary
            Keyword parameters to pass to the function

        Return
        ------
        ls : iterable of results
            each individual result is the execution of the function on a given
            subset of the data. The caller must aggregate the result accordinly
        """
        pass

    @abstractmethod
    def executeWithStart(self, descr, function, data, *args, **kwargs):
        """
        Get the result of the task directly. The difference with meth:`execute`
        comes from the function signature, which now must have a dedicated
        keyword argument "startIndex" which indicates the start index of the
        data slice on which the function is called

        Parameters
        ----------
        desc : str
            A string describing the task
        function : callable : f(...startIndex,...)
            The function to process the task. The function must be able to
            work on any subset of the data and must have a dedicated keyword
            argument "startIndex" which indicates the start index of the data
            slice on which the function is called
        data : an iterable of piece of data
            The data to process
        args : iterable
            Parameters to pass to the function
        kwargs: dictionnary
            Keyword parameters to pass to the function

        Return
        ------
        ls : iterable of results
            each individual result is the execution of the function on a given
            subset of the data. The caller must aggregate the result accordinly
        """
        pass

    def __call__(self, descr, function, data, *args, **kwargs):
        """Delegate to :meth:`execute`"""
        return self.execute(descr, function, data, *args, **kwargs)

    @abstractmethod
    def createArray(self, shape, dtype=float):
        """
        Return
        ------
        array : numpy array
            An empty (zero-filled) numpy array of given shape and dtype,
            modifiable on place by the function used with :meth:`execute`
        """
        pass

    @abstractmethod
    def clean(self, array):
        pass


class SerialExecutor(TaskExecutor):
    """
    ==============
    SerialExecutor
    ==============
    :class:`SerialExecutor` simply store the task to execute later
    """

    def __init__(self, logger=None, verbosity=10):
        TaskExecutor.__init__(self, logger, verbosity)

    def execute(self, desc, function, data, *args, **kwargs):
        self.setTask(1, desc)
        if len(args) == 0:
            if len(kwargs) == 0:
                ls = [function(data)]
            else:
                ls = [function(data, **kwargs)]
        elif len(kwargs) == 0:
            ls = [function(data, *args)]
        else:
            ls = [function(data, *args, **kwargs)]
        self.endTask()
        return ls

    def executeWithStart(self, desc, function, data, *args, **kwargs):
        self.setTask(1, desc)
        if len(args) == 0:
            if len(kwargs) == 0:
                ls = [function(data, startIndex=0)]
            else:
                ls = [function(data, startIndex=0, **kwargs)]
        elif len(kwargs) == 0:
            ls = [function(data, startIndex=0, *args)]
        else:
            ls = [function(data, startIndex=0, *args, **kwargs)]
        self.endTask()
        return ls

    def createArray(self, shape, dtype=float):
        return np.zeros(shape, dtype)

    def clean(self, array):
        pass


class ParallelExecutor(TaskExecutor):
    """
    ====================
    ParallelExecutor
    ====================
    :class:`ParallelExecutor` splits the data for multiprocessing
    """

    def __init__(self, nbParal=-1, logger=None, verbosity=0, tempFolder=None):
        TaskExecutor.__init__(self, logger, verbosity)
        self._nbParal = nbParal
        self._tmpFolder = tempFolder
        self._numpyFactory = NumpyFactory(tempFolder)

    def execute(self, desc, function, data, *args, **kwargs):
        #Splitting task
        tSplitter = TaskSplitter()
        nbJobs, splittedData, starts = tSplitter.partition(self._nbParal, data)

        #Logging
        self.setTask(1, ("Starting parallelization : "+desc))

        #Parallelization
        parallelizer = Parallel(n_jobs=nbJobs, temp_folder=self._tmpFolder,
                                verbose=self.verbosity,)

        if len(args) == 0:
            if len(kwargs) == 0:
                allData = parallelizer(delayed(function)(
                    splittedData[i]) for i in xrange(nbJobs))
            else:
                allData = parallelizer(delayed(function)(
                    splittedData[i], **kwargs) for i in xrange(nbJobs))
        elif len(kwargs) == 0:
            allData = parallelizer(delayed(function)(
                splittedData[i], *args) for i in xrange(nbJobs))
        else:
            allData = parallelizer(delayed(function)(
                splittedData[i], *args, **kwargs) for i in xrange(nbJobs))
        self.endTask()

        return allData

    def executeWithStart(self, desc, function, data, *args, **kwargs):
        #Splitting task
        tSplitter = TaskSplitter()
        nbJobs, splittedData, starts = tSplitter.partition(self._nbParal, data)

        #Logging
        self.setTask(1, ("Starting parallelization : "+desc))

        #Parallelization
        parallelizer = Parallel(n_jobs=nbJobs, temp_folder=self._tmpFolder,
                                verbose=self.verbosity,)

        if len(args) == 0:
            if len(kwargs) == 0:
                allData = parallelizer(delayed(function)(
                    splittedData[i], startIndex=starts[i])
                    for i in xrange(nbJobs))
            else:
                allData = parallelizer(delayed(function)(
                    splittedData[i], startIndex=starts[i], **kwargs)
                    for i in xrange(nbJobs))

        elif len(kwargs) == 0:
            allData = parallelizer(delayed(function)(
                splittedData[i], startIndex=starts[i], *args)
                for i in xrange(nbJobs))

        else:
            allData = parallelizer(delayed(function)(
                splittedData[i], startIndex=starts[i], *args, **kwargs)
                for i in xrange(nbJobs))

        self.endTask()

        return allData

    def createArray(self, shape, dtype=float):
        return self._numpyFactory.createArray(shape, dtype)

    def clean(self, array):
        self._numpyFactory.clean(array)

#class ParallelCoordinator(Coordinator):
#    """
#    ===================
#    ParallelCoordinator
#    ===================
#    A coordinator (see :class:`Coordinator`) for parallel computing
#    """
#    #Counts the number of instances already created
#    instanceCounter = 0
#
#    def __init__(self, coordinator, nbParal=-1, verbosity=0, tempFolder=None):
#        """
#        Construct a :class:`ParallelCoordinator`
#
#        Parameters
#        ----------
#        coordinator : :class:`Coordinator`
#            The coordinator which will execute the work in its private
#            child process
#        nbParal : int {-1, > 0} (default : -1)
#            The parallel factor. If -1, or > #cpu, the maximum factor is used
#            (#cpu)
#        verbosity : int >=0 (default : 0)
#            The verbosity level. The higher, the more information message.
#            Information message are printed on the stderr
#        tempFolder : string (directory path) (default : None)
#            The temporary folder used for memmap. If none, some default folder
#            will be use (see the :lib:`joblib` library)
#        """
#        Coordinator.__init__(self)
#        self._coordinator = coordinator
#        self._nbParal = nbParal
#        self._verbosity = verbosity
#
#        if ParallelCoordinator.instanceCounter == 0:
#            copy_reg.pickle(types.MethodType, reduceMethod)
#        ParallelCoordinator.instanceCounter += 1
#
#    def process(self, imageBuffer):
#        taskSplitter = TaskSplitter()
#        nbJobs, subImageBuffer = taskSplitter.partition(self._nbParal,
#                                                       imageBuffer)
#        #Logging
#        self.setTask(1, "Starting parallelization")
#
#        #Parallelization
#        allData = Parallel(n_jobs=nbJobs, verbose=self._verbosity)(
#            delayed(self._coordinator.process)(
#                subImageBuffer[i])
#            for i in xrange(nbJobs))
#
#        # Reduce
#        self.logMsg("Concatenating the data...", 35)
#        y = np.concatenate([y for _, y in allData])
#        X = np.vstack(X for X, _ in allData)
#        self.endTask()
#
#        return X, y

if __name__ == "__main__":
    test1 = [("A",1), ("B",2), ("C",3), ("D",4)]
    test2 = [("A",1),("B",2),("C",3),("D",4),("E",5),("F",6),("G",7),("H",8)]
    test3 = [("A",1),("B",2),("C",3),("D",4),("E",5),("F",6),("G",7),("H",8),("I",9),("J",10),("K",11),("L",12)]

    taskMan = TaskSplitter()

