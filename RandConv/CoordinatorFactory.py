# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
"""
A set of factory function to help create usual cases of coordinator
"""
import math

from FilterGenerator import (FilterGenerator, Finite3SameFilter,
                             IdPerturbatedFG, IdMaxL1DistPerturbFG,
                             StratifiedFG, OrderedMFF, SparsityDecoratorFG)
from FilterHolder import customFinite3sameFilter, customFilters
from Convolver import RGBConvolver
from SubWindowExtractor import (MultiSWExtractor, SubWindowExtractor,
                                FixTargetSWExtractor)
from NumberGenerator import (OddUniformGenerator, NumberGenerator,
                             CustomDiscreteNumberGenerator,
                             GaussianNumberGenerator)
from FeatureExtractor import ImageLinearizationExtractor, DepthCompressorILE
from Pooler import (IdentityPooler, MultiPooler, ConvMinPooler,
                    ConvAvgPooler, ConvMaxPooler, MorphOpeningPooler,
                    MorphClosingPooler)
from FastPooling import FastMWAvgPooler, FastMWMinPooler, FastMWMaxPooler
from Aggregator import AverageAggregator, MaximumAggregator, MinimumAggregator
from ConvolutionalExtractor import ConvolutionalExtractor
from Coordinator import (RandConvCoordinator, PixitCoordinator,
                         CompressRandConvCoordinator)
from Logger import StandardLogger, ProgressLogger
from Compressor import SamplerFactory, PCAFactory


__all__ = ["Const", "coordinatorRandConvFactory", "customRandConvFactory",
           "coordinatorPixitFactory", "coordinatorCompressRandConvFactory"]


class Const:
    RND_RU = "RND_RU"  # -1 (real uniform)
    RND_SET = "RND_SET"  # -2 (Discrete set with predifined probabilities)
    RND_GAUSS = "RND_GAUSS"  # (Gaussian distribution)

    FGEN_ORDERED = "FGEN_ORDERED"  # Ordered combination of others
    FGEN_CUSTOM = "FGEN_CUSTOM"  # Custom filters
    FGEN_ZEROPERT = "FGEN_ZEROPERT"  # Perturbation around origin
    FGEN_IDPERT = "FGEN_IDPERT"  # Perturbation around id filter
    FGEN_IDDIST = "FGEN_IDDIST"  # Maximum distance around id filter
    FGEN_STRAT = "FGEN_STRAT"  # Stratified scheme

    POOLING_NONE = "POOLING_NONE"  # 0
    POOLING_AGGREG_MIN = "POOLING_AGGREG_MIN"  # 1
    POOLING_AGGREG_AVG = "POOLING_AGGREG_AVG"  # 2
    POOLING_AGGREG_MAX = "POOLING_AGGREG_MAX"  # 3
    POOLING_CONV_MIN = "POOLING_MW_MIN"  # 4
    POOLING_CONV_AVG = "POOLING_MW_AVG"  # 5
    POOLING_CONV_MAX = "POOLING_MW_MAX"  # 6
    POOLING_MORPH_OPENING = "POOLING_MORPH_OPENING"  # 7
    POOLING_MORPH_CLOSING = "POOLING_MORPH_CLOSING"  # 8

    FEATEXT_ALL = "FEATEXTRACT_ALL"
    FEATEXT_SPASUB = "FEATEXTRACT_SPASUB"


def coordinatorPixitFactory(
        nbSubwindows=10,
        subwindowMinSizeRatio=0.5, subwindowMaxSizeRatio=1.,
        subwindowTargetWidth=16, subwindowTargetHeight=16,
        fixedSize=False,
        subwindowInterpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        nbJobs=-1, verbosity=10, tempFolder=None,
        random=True):
    """
    Factory method to create :class:`PixitCoordinator`

    Parameters
    ----------
    nbSubwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    subwindowMinSizeRatio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowMaxSizeRatio : float : subwindowMinSizeRatio
    <= subwindowMaxSizeRatio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowTargetWidth : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    subwindowTargetHeight : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    fixedSize : boolean (default : False)
        Whether to use fixe size subwindow. If False, subwindows are drawn
        randomly. If True, the target size is use as the subwindow size and
        only the position is drawn randomly
    subwindowInterpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    nbJobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    tempFolder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)
    random : bool (default : True)
        Whether to use randomness or use a predefined seed

    Return
    ------
        coordinator : :class:`Coordinator`
            The PixitCoordinator (possibly decorated) corresponding to the set
            of parameters
    Notes
    -----
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformely).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """

    swngSeed = 0
    #Randomness
    if random:
        swngSeed = None
    #SubWindowExtractor
    swNumGenerator = NumberGenerator(seed=swngSeed)
    if fixedSize:
        swExtractor = FixTargetSWExtractor(subwindowTargetWidth,
                                           subwindowTargetHeight,
                                           subwindowInterpolation,
                                           swNumGenerator)
    else:
        swExtractor = SubWindowExtractor(subwindowMinSizeRatio,
                                         subwindowMaxSizeRatio,
                                         subwindowTargetWidth,
                                         subwindowTargetHeight,
                                         subwindowInterpolation,
                                         swNumGenerator)

    multiSWExtractor = MultiSWExtractor(swExtractor, nbSubwindows, True)

    #FEATURE EXTRACTOR
    featureExtractor = ImageLinearizationExtractor()

    #LOGGER
    autoFlush = verbosity >= 45
    logger = ProgressLogger(StandardLogger(autoFlush=autoFlush,
                                           verbosity=verbosity))

    #COORDINATOR
    coordinator = PixitCoordinator(multiSWExtractor, featureExtractor, logger,
                                   verbosity)

    if nbJobs != 1:
        coordinator.parallelize(nbJobs, tempFolder)
    return coordinator


def getMultiPoolers(poolings, finalHeight, finalWidth):
    #Aggregator
    poolers = []
    for height, width, policy in poolings:
        if policy is Const.POOLING_NONE:
            poolers.append(IdentityPooler())
        elif policy is Const.POOLING_AGGREG_AVG:
            poolers.append(AverageAggregator(width, height,
                                             finalWidth,
                                             finalHeight))
        elif policy is Const.POOLING_AGGREG_MAX:
            poolers.append(MaximumAggregator(width, height,
                                             finalWidth,
                                             finalHeight))
        elif policy is Const.POOLING_AGGREG_MIN:
            poolers.append(MinimumAggregator(width, height,
                                             finalWidth,
                                             finalHeight))
        elif policy is Const.POOLING_CONV_MIN:
            poolers.append(FastMWMinPooler(height, width))
        elif policy is Const.POOLING_CONV_AVG:
            poolers.append(FastMWAvgPooler(height, width))
        elif policy is Const.POOLING_CONV_MAX:
            poolers.append(FastMWMaxPooler(height, width))
        elif policy is Const.POOLING_MORPH_OPENING:
            poolers.append(MorphOpeningPooler(height, width))
        elif policy is Const.POOLING_MORPH_CLOSING:
            poolers.append(MorphClosingPooler(height, width))

    return MultiPooler(poolers)


def getNumberGenerator(genType, minValue, maxValue, seed, **kwargs):
    if genType is Const.RND_RU:
        valGenerator = NumberGenerator(minValue, maxValue, seed)
    elif genType is Const.RND_SET:
        probLaw = kwargs["probLaw"]
        valGenerator = CustomDiscreteNumberGenerator(probLaw, seed)
    elif genType is Const.RND_GAUSS:
        if "outRange" in kwargs:
            outRange = kwargs["outRange"]
            valGenerator = GaussianNumberGenerator(minValue, maxValue, seed,
                                                   outRange)
        else:
            valGenerator = GaussianNumberGenerator(minValue, maxValue, seed)
    return valGenerator


def getFilterGenerator(policy, parameters, nbFilters, random=False):
    if policy == Const.FGEN_ORDERED:
        #Parameters is a list of tuples (policy, parameters)
        ls = []
        subNbFilters = int(math.ceil(nbFilters/len(parameters)))

        for subPolicy, subParameters in parameters:
            ls.append(getFilterGenerator(subPolicy, subParameters,
                                         subNbFilters, random))
        return OrderedMFF(ls, nbFilters)

    if policy is Const.FGEN_CUSTOM:
        print "Custom filters"
        return customFinite3sameFilter()

    #Parameters is a dictionary
    valSeed = None
    sizeSeed = None
    shufflingSeed = None
    perturbationSeed = None
    cellSeed = None
    sparseSeed = 5
    if random:
        valSeed = 1
        sizeSeed = 2
        shufflingSeed = 3
        perturbationSeed = 4
        cellSeed = 5
        sparseSeed = 6

    minSize = parameters["minSize"]
    maxSize = parameters["maxSize"]
    sizeGenerator = OddUniformGenerator(minSize, maxSize, seed=sizeSeed)

    minVal = parameters["minVal"]
    maxVal = parameters["maxVal"]
    valGen = parameters["valGen"]
    valGenerator = getNumberGenerator(valGen, minVal, maxVal,
                                      valSeed, **parameters)

    normalization = None
    if "normalization" in parameters:
        normalization = parameters["normalization"]

    if policy is Const.FGEN_ZEROPERT:
        print "Zero perturbation filters"
        baseFilterGenerator = FilterGenerator(valGenerator, sizeGenerator,
                                              normalisation=normalization)

    elif policy is Const.FGEN_IDPERT:
        print "Id perturbation filters"
        baseFilterGenerator = IdPerturbatedFG(valGenerator, sizeGenerator,
                                              normalisation=normalization)
    elif policy is Const.FGEN_IDDIST:
        print "Id distance filters"
        maxDist = parameters["maxDist"]
        baseFilterGenerator = IdMaxL1DistPerturbFG(valGenerator, sizeGenerator,
                                                   maxDist,
                                                   normalisation=normalization,
                                                   shufflingSeed=shufflingSeed)
    elif policy is Const.FGEN_STRAT:
        print "Stratified filters"
        nbCells = parameters["strat_nbCells"]
        minPerturbation = 0
        if "minPerturbation" in parameters:
            minPerturbation = parameters["minPerturbation"]
        maxPerturbation = 1
        if "maxPerturbation" in parameters:
            maxPerturbation = parameters["maxPerturbation"]
        perturbationGenerator = getNumberGenerator(valGen,
                                                   minPerturbation,
                                                   maxPerturbation,
                                                   perturbationSeed)
        baseFilterGenerator = StratifiedFG(minVal, maxVal, nbCells,
                                           perturbationGenerator,
                                           sizeGenerator,
                                           normalisation=normalization,
                                           cellSeed=cellSeed)

    if "sparseProb" in parameters:
        print "Adding sparcity"
        sparseProb = parameters["sparseProb"]
        baseFilterGenerator = SparsityDecoratorFG(baseFilterGenerator,
                                                  sparseProb,
                                                  sparseSeed)

    print "Returning filters"
    return Finite3SameFilter(baseFilterGenerator, nbFilters)


def getFeatureExtractor(policy, **kwargs):
    if policy is Const.FEATEXT_SPASUB:
        nbCol = kwargs.get("nbCol", 2)
        return DepthCompressorILE(nbCol)

    else:  # Suupose Const.FEATEXT_ALL
        return ImageLinearizationExtractor()

#TODO : include in randconv : (Const.FEATEXT_ALL, {}), (Const.FEATEXT_SPASUB, {"nbCol":2})
def coordinatorRandConvFactory(
        nbFilters=5,
        filterPolicy=(Const.FGEN_ZEROPERT,
                      {"minSize": 2, "maxSize": 32, "minVal": -1, "maxVal": 1,
                       "valGen": Const.RND_RU,
                       "normalization": FilterGenerator.NORMALISATION_MEANVAR}),
        poolings=[(3, 3, Const.POOLING_AGGREG_AVG)],
        extractor=(Const.FEATEXT_ALL, {}),
        nbSubwindows=10,
        subwindowMinSizeRatio=0.5, subwindowMaxSizeRatio=1.,
        subwindowTargetWidth=16, subwindowTargetHeight=16,
        subwindowInterpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        includeOriginalImage=False,
        nbJobs=-1, verbosity=10, tempFolder=None,
        random=True):
    """
    Factory method to create :class:`RandConvCoordinator` tuned for RGB images

    Parameters
    ----------
    nbFilters : int >= 0 (default : 5)
        The number of filter

    filterPolicy : pair (policyType, parameters)
        policyType : one of Const.FGEN_*
            The type of filter generation policy to use
        parameters : dict
            The parameter dictionnary to forward to :func:`getFilterGenerator`

    poolings : iterable of triple (height, width, policy) (default :
    [(3, 3, Const.POOLING_AGGREG_AVG)])
        A list of parameters to instanciate the according :class:`Pooler`
        height : int > 0
            the height of the neighborhood window
        width : int > 0
            the width of the neighborhood window
        policy : int in {Const.POOLING_NONE, Const.POOLING_AGGREG_MIN,
    Const.POOLING_AGGREG_AVG, Const.POOLING_AGGREG_MAX,
    Const.POOLING_CONV_MIN, Const.POOLING_CONV_AVG, Const.POOLING_CONV_MAX}

    nbSubwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    subwindowMinSizeRatio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowMaxSizeRatio : float : subwindowMinSizeRatio
    <= subwindowMaxSizeRatio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowTargetWidth : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    subwindowTargetHeight : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    subwindowInterpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    includeOriginalImage : boolean (default : False)
        Whether or not to include the original image in the subwindow
        extraction process

    nbJobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    tempFolder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)

    random : bool (default : True)
        Whether to use randomness or use a predefined seed

    Return
    ------
        coordinator : :class:`Coordinator`
            The RandConvCoordinator corresponding to the
            set of parameters

    Notes
    -----
    - Filter generator
        Base instance of :class:`Finite3SameFilter` with a base instance of
        :class:`NumberGenerator` for the values and
        :class:`OddUniformGenerator` for the sizes
    - Filter size
        The filter are square (same width as height)
    - Convolver
        Base instance of :class:`RGBConvolver`
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformely).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """
    #RANDOMNESS
    swngSeed = None
    if random is False:
        swngSeed = 0

    #CONVOLUTIONAL EXTRACTOR
    #Filter generator
    #Type/policy parameters, #filters, random
    filterPolicyType, filterPolicyParam = filterPolicy
    filterGenerator = getFilterGenerator(filterPolicyType, filterPolicyParam,
                                         nbFilters, random)

    #Convolver
    convolver = RGBConvolver()

    #Aggregator
    multiPooler = getMultiPoolers(poolings, subwindowTargetHeight,
                                  subwindowTargetWidth)

    #SubWindowExtractor
    swNumGenerator = NumberGenerator(seed=swngSeed)
    swExtractor = SubWindowExtractor(subwindowMinSizeRatio,
                                     subwindowMaxSizeRatio,
                                     subwindowTargetWidth,
                                     subwindowTargetHeight,
                                     subwindowInterpolation, swNumGenerator)

    multiSWExtractor = MultiSWExtractor(swExtractor, nbSubwindows, False)

    #ConvolutionalExtractor
    convolutionalExtractor = ConvolutionalExtractor(filterGenerator,
                                                    convolver,
                                                    multiSWExtractor,
                                                    multiPooler,
                                                    includeOriginalImage)
    #FEATURE EXTRACTOR
    featureExtractor = getFeatureExtractor(extractor[0], extractor[1])

    #LOGGER
    autoFlush = verbosity >= 40
    logger = ProgressLogger(StandardLogger(autoFlush=autoFlush,
                                           verbosity=verbosity))
    #COORDINATOR
    coordinator = RandConvCoordinator(convolutionalExtractor, featureExtractor,
                                      logger, verbosity)

    if nbJobs != 1:
        coordinator.parallelize(nbJobs, tempFolder)
    return coordinator


def customRandConvFactory(
        poolings=[(3, 3, Const.POOLING_AGGREG_AVG)],
        nbSubwindows=10,
        subwindowMinSizeRatio=0.5, subwindowMaxSizeRatio=1.,
        subwindowTargetWidth=16, subwindowTargetHeight=16,
        subwindowInterpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        includeOriginalImage=False,
        nbJobs=-1, verbosity=10, tempFolder=None,
        random=True):
    """
    Factory method to create :class:`RandConvCoordinator` tuned for RGB images
    using predefined well-known filters

    Parameters
    ----------
    poolings : iterable of triple (height, width, policy) (default :
    [(3, 3, Const.POOLING_AGGREG_AVG)])
        A list of parameters to instanciate the according :class:`Pooler`
        height : int > 0
            the height of the neighborhood window
        width : int > 0
            the width of the neighborhood window
        policy : int in {Const.POOLING_NONE, Const.POOLING_AGGREG_MIN,
    Const.POOLING_AGGREG_AVG, Const.POOLING_AGGREG_MAX,
    Const.POOLING_CONV_MIN, Const.POOLING_CONV_AVG, Const.POOLING_CONV_MAX}

    nbSubwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    subwindowMinSizeRatio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowMaxSizeRatio : float : subwindowMinSizeRatio
    <= subwindowMaxSizeRatio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowTargetWidth : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    subwindowTargetHeight : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    subwindowInterpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    includeOriginalImage : boolean (default : False)
        Whether or not to include the original image in the subwindow
        extraction process

    nbJobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    tempFolder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)

    random : bool (default : True)
        Whether to use randomness or use a predefined seed

    Return
    ------
        coordinator : :class:`Coordinator`
            The RandConvCoordinator corresponding to the
            set of parameters

    Notes
    -----
    - Convolver
        Base instance of :class:`RGBConvolver`
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformly).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """
    #RANDOMNESS
    swngSeed = 0
    if random is None:
        swngSeed = None

    #CONVOLUTIONAL EXTRACTOR
    filterGenerator = customFinite3sameFilter()

    #Convolver
    convolver = RGBConvolver()

    #Aggregator
    poolers = []
    for height, width, policy in poolings:
        if policy == Const.POOLING_NONE:
            poolers.append(IdentityPooler())
        elif policy == Const.POOLING_AGGREG_AVG:
            poolers.append(AverageAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_AGGREG_MAX:
            poolers.append(MaximumAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_AGGREG_MIN:
            poolers.append(MinimumAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_CONV_MIN:
            poolers.append(ConvMinPooler(height, width))
        elif policy == Const.POOLING_CONV_AVG:
            poolers.append(ConvAvgPooler(height, width))
        elif policy == Const.POOLING_CONV_MAX:
            poolers.append(ConvMaxPooler(height, width))

    multiPooler = getMultiPoolers(subwindowTargetHeight, subwindowTargetWidth)

    #SubWindowExtractor
    swNumGenerator = NumberGenerator(seed=swngSeed)
    swExtractor = SubWindowExtractor(subwindowMinSizeRatio,
                                     subwindowMaxSizeRatio,
                                     subwindowTargetWidth,
                                     subwindowTargetHeight,
                                     subwindowInterpolation, swNumGenerator)

    multiSWExtractor = MultiSWExtractor(swExtractor, nbSubwindows, False)

    #ConvolutionalExtractor
    convolutionalExtractor = ConvolutionalExtractor(filterGenerator,
                                                    convolver,
                                                    multiSWExtractor,
                                                    multiPooler,
                                                    includeOriginalImage)
    #FEATURE EXTRACTOR
    featureExtractor = ImageLinearizationExtractor()

    #LOGGER
    autoFlush = verbosity >= 45
    logger = ProgressLogger(StandardLogger(autoFlush=autoFlush,
                                           verbosity=verbosity))
    #COORDINATOR
    coordinator = RandConvCoordinator(convolutionalExtractor, featureExtractor,
                                      logger, verbosity)

    if nbJobs != 1:
        coordinator.parallelize(nbJobs, tempFolder)
    return coordinator


def coordinatorCompressRandConvFactory(
        nbFilters=5,
        filterMinVal=-1, filterMaxVal=1,
        filterMinSize=1, filterMaxSize=17,
        filterNormalisation=FilterGenerator.NORMALISATION_MEANVAR,
        poolings=[(3, 3, Const.POOLING_AGGREG_AVG)],
        nbSubwindows=10,
        subwindowMinSizeRatio=0.5, subwindowMaxSizeRatio=1.,
        subwindowTargetWidth=16, subwindowTargetHeight=16,
        subwindowInterpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        includeOriginalImage=False,
        compressorType="Sampling", nbCompressedFeatures=1,
        compressOriginalImage=True,
        nbJobs=-1, verbosity=10, tempFolder=None,
        random=True):
    """
    Factory method to create :class:`RandConvCoordinator` tuned for RGB images

    Parameters
    ----------
    nbFilters : int >= 0 (default : 5)
        The number of filter
    filterMinVal : float (default : -1)
        The minimum value of a filter component
    filterMaxVal : float : filterMinVal <= filterMaxVal (default : 1)
        The maximum value of a filter component
    filterMinSize : int >= 0 : odd number (default : 1)
        The minimum size of a filter
    filterMaxSize : int >= 0 : odd number s.t.  filterMinSize <= filterMaxSize
    (default : 1)
        The maximum size of a filter
    filterNormalisation : int (default : FilterGenerator.NORMALISATION_MEANVAR)
        The filter normalisation policy. See also :class:`FilterGenerator`

    poolings : iterable of triple (height, width, policy) (default :
    [(3, 3, Const.POOLING_AGGREG_AVG)])
        A list of parameters to instanciate the according :class:`Pooler`
        height : int > 0
            the height of the neighborhood window
        width : int > 0
            the width of the neighborhood window
        policy : int in {Const.POOLING_NONE, Const.POOLING_AGGREG_MIN,
    Const.POOLING_AGGREG_AVG, Const.POOLING_AGGREG_MAX,
    Const.POOLING_CONV_MIN, Const.POOLING_CONV_AVG, Const.POOLING_CONV_MAX}

    nbSubwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    subwindowMinSizeRatio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowMaxSizeRatio : float : subwindowMinSizeRatio
    <= subwindowMaxSizeRatio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowTargetWidth : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    subwindowTargetHeight : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    subwindowInterpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    includeOriginalImage : boolean (default : False)
        Whether or not to include the original image in the subwindow
        extraction process

    compressorType : str (default : "Sampling")
        The type of compressor to use. One of the {"Sampling", "PCA"}
    nbCompressedFeatures : int > 0 (default : 1)
        The number features per filter after compression. It should be
        inferior to the number of features per filter traditionally output by
        the :class:`randConvCoordinator`
    compressOriginalImage : boolean (default : True)
        Whether to compress the original (if included) as well as the other
        features

    nbJobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    tempFolder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)

    random : bool (default : True)
        Whether to use randomness or use a predefined seed
    Return
    ------
        coordinator : :class:`Coordinator`
            The RandConvCoordinator (possibly decorated) corresponding to the
            set of parameters

    Notes
    -----
    - Filter generator
        Base instance of :class:`Finite3SameFilter` with a base instance of
        :class:`NumberGenerator` for the values and
        :class:`OddUniformGenerator` for the sizes
    - Filter size
        The filter are square (same width as height)
    - Convolver
        Base instance of :class:`RGBConvolver`
    - Aggregator
        Base instance of :class:`AverageAggregator`
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformely).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """

    #RANDOMNESS
    swngSeed = 0
    filtValGenSeed = 1
    filtSizeGenSeed = 2
    if random is None:
        swngSeed = None
        filtValGenSeed = None
        filtSizeGenSeed = None

    #CONVOLUTIONAL EXTRACTOR
    #Filter generator
    filterValGenerator = NumberGenerator(filterMinVal, filterMaxVal,
                                         seed=filtValGenSeed)
    filterSizeGenerator = OddUniformGenerator(filterMinSize, filterMaxSize,
                                              seed=filtSizeGenSeed)
    baseFilterGenerator = FilterGenerator(filterValGenerator,
                                          filterSizeGenerator,
                                          normalisation=filterNormalisation)
    filterGenerator = Finite3SameFilter(baseFilterGenerator, nbFilters)

    #Convolver
    convolver = RGBConvolver()

    #Aggregator
    poolers = []
    for height, width, policy in poolings:
        if policy == Const.POOLING_NONE:
            poolers.append(IdentityPooler())
        elif policy == Const.POOLING_AGGREG_AVG:
            poolers.append(AverageAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_AGGREG_MAX:
            poolers.append(MaximumAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_AGGREG_MIN:
            poolers.append(MinimumAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_CONV_MIN:
            poolers.append(ConvMinPooler(height, width))
        elif policy == Const.POOLING_CONV_AVG:
            poolers.append(ConvAvgPooler(height, width))
        elif policy == Const.POOLING_CONV_MAX:
            poolers.append(ConvMaxPooler(height, width))

    multiPooler = MultiPooler(poolers)

    #SubWindowExtractor
    swNumGenerator = NumberGenerator(seed=swngSeed)
    swExtractor = SubWindowExtractor(subwindowMinSizeRatio,
                                     subwindowMaxSizeRatio,
                                     subwindowTargetWidth,
                                     subwindowTargetHeight,
                                     subwindowInterpolation, swNumGenerator)

    multiSWExtractor = MultiSWExtractor(swExtractor, nbSubwindows, False)

    #ConvolutionalExtractor
    convolutionalExtractor = ConvolutionalExtractor(filterGenerator,
                                                    convolver,
                                                    multiSWExtractor,
                                                    multiPooler,
                                                    includeOriginalImage)
    #FEATURE EXTRACTOR
    featureExtractor = ImageLinearizationExtractor()

    #LOGGER
    autoFlush = verbosity >= 45
    logger = ProgressLogger(StandardLogger(autoFlush=autoFlush,
                                           verbosity=verbosity))

    #COMPRESSOR
    if compressorType == "PCA":
        compressorFactory = PCAFactory(nbCompressedFeatures)
    else:
        #"Sampling"
        compressorFactory = SamplerFactory(nbCompressedFeatures)

    #COORDINATOR
    coordinator = CompressRandConvCoordinator(convolutionalExtractor,
                                              featureExtractor,
                                              compressorFactory,
                                              compressOriginalImage,
                                              logger, verbosity)

    if nbJobs != 1:
        coordinator.parallelize(nbJobs, tempFolder)
    return coordinator


if __name__ == "__main__":

    from CifarLoader import CifarLoader
    from ImageBuffer import ImageBuffer

    path = "data_batch_1"
    imgBuffer = CifarLoader(path, outputFormat=ImageBuffer.NUMPY_FORMAT)

    #coord = coordinatorRandConvFactory(verbosity=50)
    coord = coordinatorRandConvFactory(nbJobs=1, verbosity=0)

    X, y = coord.process(imgBuffer[0:10])
    print X.shape, len(y)
