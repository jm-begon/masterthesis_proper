# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 16 2014
"""
A Logging scheme
"""

import sys
import os
import time
from abc import ABCMeta, abstractmethod

__all__ = ["Logger", "StandardLogger", "FileLogger", "ProgressLogger",
           "Progressable", "formatDuration", "formatSize"]


def formatDuration(duration):
    """
    Format the duration as string

    Parameters
    ----------
    duration : float
        a duration in seconds
    Return
    ------
    formated : str
        the given duration formated
    """
    sec = duration % 60
    excess = int(duration) // 60  # minutes
    res = str(sec) + "s"
    if excess == 0:
        return res
    minutes = excess % 60
    excess = excess // 60  # hours
    res = str(minutes) + "m " + res
    if excess == 0:
        return res
    hour = excess % 24
    excess = excess // 24  # days
    res = str(hour) + "h " + res
    print "hours", hour
    if excess == 0:
        return res
    res = str(excess)+"d " + res
    return res


def formatSize(nbBytes):
    for x in ['bytes', 'kB', 'MB', 'GB']:
        if nbBytes < 1000.0:
            return "%3.1f %s" % (nbBytes, x)
        nbBytes /= 1000.0
    return "%3.1f %s" % (nbBytes, 'TB')


class Logger:
    """
    ======
    Logger
    ======
    A abstract base class for the logging facility

    Attributes
    ----------
    verbosity : int [0,50] (default : 50)
        the degree of verbosity (the more, the more verbose)
    """
    __metaclass__ = ABCMeta

    def __init__(self, verbosity=50):
        """
        Construct a :class:`Logger`

        Parameters
        ----------
        verbosity : int [0,50] (default : 50)
            the degree of verbosity (the more, the more verbose)
        """
        self.setVerbosity(verbosity)

    def setVerbosity(self, verbosity):
        """
        Set the verbosity level

        Parameters
        ----------
        verbosity : int [0,50] (default : 50)
            the degree of verbosity (the more, the more verbose)
        """
        if verbosity < 0:
            verbosity = 0
        self.verbosity = verbosity

    @abstractmethod
    def logMsg(self, message, minVerb=1):
        """
        Log the given message

        Parameters
        ----------
        message : str
            The message string to log
        minVerb : int >= 0 (default : 1)
            The minimum verbosity required to log this message
        """
        pass

    def logError(self, errorMsg, minVerb=0):
        """
        Log the error message

        Parameters
        ----------
        errorMsg : str
            The error message to log
        minVerb : int >= 0 (default : 0)
            The minimum verbosity required to log this message
        """
        self.logMsg(errorMsg)

    def logSize(self, message, nbBytes, minVerb=1):
        self.logMsg(message+formatSize(nbBytes), minVerb=minVerb)

    def appendLN(self, msg):
        """
        Append line ending (os.linesep) if necessary.

        Paramters
        ---------
        msg : str
            a message

        Return
        ------
        msgLn : str
            the same message with a new line added if necessary
        """
        if not msg.endswith(os.linesep):
            msg += os.linesep
        return msg


class VoidLogger(Logger):
    def __init__(self, verbosity=10):
        Logger.__init__(self, verbosity)

    def logMsg(self, msg, minVerb=1):
        pass


class StandardLogger(Logger):
    def __init__(self, autoFlush=False, verbosity=50):
        """
        Construct a :class:`StandardLogger` (with stdout and stderr)

        Parameters
        ----------
        autoFlush : boolean (Default : False)
            Whether to flush after each log or not
        verbosity : int [0,50] (default : 50)
            the degree of verbosity (the more, the more verbose)
        """
        Logger.__init__(self, verbosity)
        self._autoFlush = autoFlush

    def logMsg(self, msg, minVerb=1):
        """Overload"""
        if self.verbosity >= minVerb:
            msg = self.appendLN(msg)
            sys.stdout.write(msg)
            if self._autoFlush:
                sys.stdout.flush()

    def logError(self, msg, minVerb=0):
        """Overload"""
        if self.verbosity >= minVerb:
            msg = self.appendLN(msg)
            sys.stderr.write(msg)
            if self._autoFlush:
                sys.stderr.flush()


class FileLogger(Logger):
    def __init__(self, outFile=sys.stdout, errFile=sys.stderr,
                 autoFlush=False, verbosity=50):
        """
        Construct a :class:`FileLogger`

        Parameters
        ----------
        outFile : file
            the output file to write to. It can also be another class which
            supports wirte and flush operations.
        errFile : file
            the output file to write error to. It can also be another class
            which supports wirte and flush operations.
        autoFlush : boolean (Default : False)
            Whether to flush after each log or not
        verbosity : int [0,50] (default : 50)
            the degree of verbosity (the more, the more verbose)
        """
        Logger.__init__(self, verbosity)
        self._str = outFile  # stream
        self._err = errFile
        self._autoFlush = autoFlush

    def logMsg(self, msg, minVerb=1):
        """Overload"""
        if self.verbosity >= minVerb:
            msg = self.appendLN(msg)
            self._str.write(msg)
            if self._autoFlush:
                self._str.flush()

    def logError(self, msg, minVerb=0):
        """Overload"""
        if self.verbosity >= minVerb:
            msg = self.appendLN(msg)
            self._err.write(msg)
            if self._autoFlush:
                self._err.flush()

    def appendLN(self, msg):
        if not msg.endswith("\n"):
            msg += "\n"


class ProgressableTask:
    """
    ================
    ProgressableTask
    ================
    A :class:`ProgressableTask` is a task whose progression can n steps.
    Once the progression hits the max, the task status goes
    from `ProgressableTask.RUNNING` to `ProgressableTask.DONE`
    """
    nbTasks = 0

    RUNNING = 1
    DONE = 2

    def __init__(self, nbStep, name=""):
        """
        Construct a :class:`ProgressableTask`

        Parameters
        ----------
        nbStep : int
            The number of step before completion
        name : str
            The name of the taks
        """
        self.name = name
        self.id = ProgressableTask.nbTasks
        ProgressableTask.nbTasks += 1

        self._nbStep = nbStep
        self._progress = 0
        self._lastReset = 0
        self._status = ProgressableTask.RUNNING
        self._endTime = None
        self._startTime = time.time()

    def getNbSteps(self):
        """TODO"""
        return self._nbStep

    def update(self, progress):
        """
        Update progress (if task is running)

        Parameters
        ----------
        progress : int
            the new progression score

        Return
        ------
        done : boolean
            True if the task is completed, False otherwise
        """
        if self._status == ProgressableTask.DONE:
            return True
        self._progress = progress
        if progress >= self._nbStep-1:
            self._status = ProgressableTask.DONE
            self._endTime = time.time()
            return True
        else:
            return False

    def reset(self):
        """TODO"""
        self._lastReset = self._progress

    def lastProgress(self):
        """
        Return
        ------
        progressPercentage : int
            The last percentage of progress
        """
        return int((100*(self._progress - self._lastReset)/self._nbStep) + .5)

    def progressAsString(self):
        """
        Return
        ------
        progress : str
            "progress/nbSteps"
        """
        return str(self._progress)+"/"+str(self._nbStep)

    def duration(self):
        """
        Return
        ------
        duration : float
            the duration of taks in seconds (up to now if still running,
            up to completion if completed)
        """
        if self._status == ProgressableTask.DONE:
            return self._endTime - self._startTime
        else:
            return time.time() - self._startTime


class ProgressLogger(Logger):
    """
    ==============
    ProgressLogger
    ==============
    A :class:`Logger` decorator which can also log progress.
    See :class:`ProgressableTask`
    """
    TASK_CREATION_VERBLVL = 8
    TASK_COMPLETION_VERBLVL = TASK_CREATION_VERBLVL  # 8
    TASK_25_PROGRESS_VERBLVL = TASK_CREATION_VERBLVL  # 8
    TASK_10_PROGRESS_VERBLVL = min((2.5*TASK_25_PROGRESS_VERBLVL, 50))  # 20
    TASK_5_PROGRESS_VERBLVL = min((2*TASK_10_PROGRESS_VERBLVL, 50))  # 40
    TASK_1_PROGRESS_VERBLVL = min((5*TASK_10_PROGRESS_VERBLVL, 50))  # 50

    def __init__(self, logger):
        """
        Construct a :class:`ProgressLogger`

        Parameters
        ----------
        Logger : :class:`Logger`
            The decorated :class:`Logger`. Verbosity is deduced from that
            logger.
        """
        Logger.__init__(self, logger.verbosity)
        self._decoLog = logger
        self._dictTask = {}

    def logMsg(self, msg, minVerb=1):
        """Overload"""
        self._decoLog.logMsg(msg, minVerb)

    def logError(self, msg, minVerb=0):
        """Overload"""
        self._decoLog.logError(msg, minVerb)

    def _logProgress(self, task, msg):
        """
        Log the progress message

        Parameters
        ----------
        taskId : int
            The id of the task
        msg : str
            The progression message to log
        """
        loggingMsg = ("Task " + str(task.id) + " '" + task.name + "' : " +
                      msg)
        self.logMsg(loggingMsg)

    def addTask(self, nbSteps, name=""):
        """
        Add a new task

        Parameters
        ----------
        nbStep : int
            The number of step before completion
        name : str
            The name of the taks
        Return
        ------
        taskId : int
            The id of the task
        """
        task = ProgressableTask(nbSteps, name)
        taskId = task.id
        self._dictTask.update({taskId: task})
        #Logging the message
        if self.verbosity >= ProgressLogger.TASK_CREATION_VERBLVL:
            self._logProgress(task,
                              "Creation " + " (" + str(nbSteps) + " steps)")
        return taskId

    def updateProgress(self, taskId, progress):
        """
        Log progress (if task is running)

        Parameters
        ----------
        taskId : int
            The id of the task
        progress : int
            The new progression score
        """
        task = self._dictTask[taskId]
        if task.update(progress):
            del self._dictTask[taskId]

            #Logging the message
            if self.verbosity >= ProgressLogger.TASK_COMPLETION_VERBLVL:
                duration = formatDuration(task.duration())
                self._logProgress(task, "Completion in " + duration)
        else:
            #Logging the message if necessary
            percProg = task.lastProgress()
            if (self.verbosity >= ProgressLogger.TASK_1_PROGRESS_VERBLVL
                    and percProg >= 1):
                task.reset()
                self._logProgress(task, task.progressAsString())
                return
            if (self.verbosity >= ProgressLogger.TASK_5_PROGRESS_VERBLVL
                    and percProg >= 5):
                task.reset()
                self._logProgress(task, task.progressAsString())
                return
            if (self.verbosity >= ProgressLogger.TASK_10_PROGRESS_VERBLVL
                    and percProg >= 10):
                task.reset()
                self._logProgress(task, task.progressAsString())
                return
            if (self.verbosity >= ProgressLogger.TASK_25_PROGRESS_VERBLVL
                    and percProg >= 25):
                task.reset()
                self._logProgress(task, task.progressAsString())
                return


class Progressable(Logger):
    """
    ============
    Progressable
    ============

    A superclass for object which would like to monitor progress on maximum
    one task at a time
    """
    def __init__(self, progressLogger=None, verbosity=10):
        """
        Construct a :class:`Progressable` instance

        Parameters
        ----------
        progressLogger : :class:`ProgressLogger` (default : None)
            the object to report to. Can be None (nothing is reported).
            If None, a :class:`StandardLogger` is provided
        verbosity : int [0,50] (default : 10)
            the degree of verbosity (the higher, the more verbose)
        """
        Logger.__init__(self, verbosity)
        if progressLogger is None:
            progressLogger = StandardLogger(verbosity)
        self.setLogger(progressLogger)
        self._task = None

    def setLogger(self, progressLogger):
        """
        Set the logger

        Parameters
        ----------
        progressLogger : :class:`ProgressLogger`
            the object to report to. Can be None (nothing is reported)

        Note
        ----
        The verbosity level is deduced from that logger
        """
        self._logger = progressLogger
        if progressLogger is not None:
            self.setVerbosity(progressLogger.verbosity)

    def setTask(self, nbSteps, name=None):
        """
        Set the task whose progress to monitor

        Parameters
        ----------
        nbStep : int
            The number of step before completion
        name : str (default : None)
            The name of the taks. If None, a name will be given to the task
        """
        if self._logger is None:
            return
        if name is None:
            name = str(self)
        self._task = self._logger.addTask(nbSteps, name)

    def updateTaskProgress(self, progress):
        """
        Log progress (if task is running)

        Parameters
        ----------
        progress : int
            The new progression score
        """
        if self._logger is None:
            return
        if self._task is not None:
            self._logger.updateProgress(self._task, progress)

    def endTask(self):
        """
        Ends the current task
        """
        if self._task is not None:
            self.updateTaskProgress(sys.maxsize)

    def logMsg(self, msg, minVerb=1):
        """Overload"""
        if self._logger is None:
            return
        self._logger.logMsg(str(self)+" : "+msg, minVerb)

    def logError(self, msg, minVerb=0):
        """Overload"""
        if self._logger is None:
            return
        self._logger.logError(str(self)+" : "+msg, minVerb)

if __name__ == "__main__":
#    import random as rd
#
#    verbosity = 20
#    t1max = 100
#    t2max = 89
#    t3max = 10
#
#    fLog = FileLogger(verbosity=verbosity)
#    logger = ProgressLogger(fLog)
#
#    id1 = logger.addTask(t1max, "t1")
#    id2 = logger.addTask(t2max, "t2")
#
#    t1 = 0
#    t2 = 0
#
#    for i in xrange(t1max+t2max-1):
#        if t1 == t1max-1 and t2 == t2max-1:
#            break
#        if t1 == t1max-1:
#            t2 += 1
#            logger.updateProgress(id2, t2)
#        elif t2 == t2max-1:
#            t1 += 1
#            logger.updateProgress(id1, t1)
#        else:
#            if rd.random() > .5:
#                t2 += 1
#                logger.updateProgress(id2, t2)
#            else:
#                t1 += 1
#                logger.updateProgress(id1, t1)
#
#    id3 = logger.addTask(t3max, "t3")
#    for i in xrange(t3max):
#        logger.updateProgress(id3, i)
#
#    print "=========DONE========"
    pass
