"""
`distpy` is a `Python` package designed to represent, draw from, evaluate, and
plot many different kinds of distributions.

**File**: $DISTPY/distpy/\\_\\_init\\_\\_.py  
**Author**: Keith Tauscher  
**Update date**: 14 May 2021
"""
from distpy.util import create_hdf5_dataset, get_hdf5_value, HDF5Link,\
    save_dictionary, load_dictionary, Savable, Loadable, bool_types,\
    int_types, float_types, real_numerical_types, complex_numerical_types,\
    numerical_types, sequence_types, univariate_histogram,\
    confidence_contour_2D, bivariate_histogram, triangle_plot, Expression,\
    SparseSquareBlockDiagonalMatrix
from distpy.transform import Transform, NullTransform, BoxCoxTransform,\
    LogTransform, ArsinhTransform, ExponentialTransform, Exp10Transform,\
    Log10Transform, PowerTransform, SineTransform, ArcsinTransform,\
    LogisticTransform, AffineTransform, ReciprocalTransform,\
    ExponentiatedTransform, LoggedTransform, SumTransform, ProductTransform,\
    CompositeTransform, castable_to_transform, cast_to_transform,\
    load_transform_from_hdf5_group, load_transform_from_hdf5_file,\
    invert_transform, TransformList, TransformSet
from distpy.distribution import Distribution, WindowedDistribution,\
    BetaDistribution, BernoulliDistribution, BinomialDistribution,\
    ChiSquaredDistribution, DoubleSidedExponentialDistribution,\
    EllipticalUniformDistribution, ExponentialDistribution, GammaDistribution,\
    SechDistribution, SechSquaredDistribution, GaussianDistribution,\
    SparseGaussianDistribution, GeometricDistribution, GriddedDistribution,\
    ParallelepipedDistribution, PoissonDistribution,\
    KroneckerDeltaDistribution, UniformDistribution,\
    TruncatedGaussianDistribution, InfiniteUniformDistribution,\
    UniformConditionDistribution, WeibullDistribution, LinkedDistribution,\
    SequentialDistribution, DirectionDistribution,\
    UniformDirectionDistribution, GeneralizedParetoDistribution,\
    GaussianDirectionDistribution, LinearDirectionDistribution,\
    UniformTriangulationDistribution, DiscreteUniformDistribution,\
    CustomDiscreteDistribution, DeterministicDistribution, DistributionSum,\
    DistributionSet, DistributionList, load_distribution_from_hdf5_group,\
    load_distribution_from_hdf5_file, DistributionHarmonizer
from distpy.jumping import JumpingDistribution, GaussianJumpingDistribution,\
    SourceDependentGaussianJumpingDistribution, UniformJumpingDistribution,\
    TruncatedGaussianJumpingDistribution, BinomialJumpingDistribution,\
    AdjacencyJumpingDistribution, GridHopJumpingDistribution,\
    SourceIndependentJumpingDistribution,\
    LocaleIndependentJumpingDistribution, JumpingDistributionSum,\
    JumpingDistributionList, JumpingDistributionSet,\
    load_jumping_distribution_from_hdf5_group,\
    load_jumping_distribution_from_hdf5_file, MetropolisHastingsSampler

util_class_names = ['Expression', 'SparseSquareBlockDiagonalMatrix']
util_class_names =\
    ['distpy.util.{0!s}.{0!s}'.format(name) for name in util_class_names]

transform_class_names = ['', 'Affine', 'Arcsin', 'Arsinh', 'BoxCox',\
    'Composite', 'Exp10', 'Exponential', 'Exponentiated', 'Log', 'Log10',\
    'Logged', 'Logistic', 'Null', 'Power', 'Product', 'Reciprocal', 'Sine',\
    'Sum']
transform_class_names = ['TransformSet', 'TransformList'] +\
    ['{!s}Transform'.format(name) for name in transform_class_names]
transform_class_names = ['distpy.transform.{0!s}.{0!s}'.format(name)\
    for name in transform_class_names]

distribution_class_names = ['', 'Bernoulli', 'Beta', 'Binomial', 'ChiSquared',\
    'CustomDiscrete', 'Deterministic', 'Direction', 'DiscreteUniform',\
    'DoubleSidedExponential', 'EllipticalUniform', 'Exponential', 'Gamma',\
    'GaussianDirection', 'Gaussian', 'GeneralizedPareto', 'Geometric',\
    'Gridded', 'InfiniteUniform', 'KroneckerDelta', 'LinearDirection',\
    'Linked', 'Parallelepiped', 'Poisson', 'Sech', 'SechSquared',\
    'Sequential', 'SparseGaussian', 'TruncatedGaussian', 'UniformCondition',\
    'UniformDirection', 'Uniform', 'UniformTriangulation', 'Weibull',\
    'Windowed']
distribution_class_names = ['DistributionList', 'DistributionSet',\
    'DistributionSum', 'DistributionHarmonizer'] +\
    ['{!s}Distribution'.format(name) for name in distribution_class_names]
distribution_class_names = ['distpy.distribution.{0!s}.{0!s}'.format(name)\
    for name in distribution_class_names]

class_names =\
    util_class_names + transform_class_names + distribution_class_names

# init not included in magic_names because __init__ is automatically documented
# hash not included because it appears automatically
magic_names = ['new', 'del', 'repr', 'str', 'bytes', 'format', 'lt', 'le',\
    'eq', 'ne', 'gt', 'ge', 'bool', 'getattr', 'getattribute', 'setattr',\
    'delattr', 'dir', 'get', 'set', 'delete', 'set_name', 'slots',\
    'init_subclass', 'class_getitem', 'call', 'len', 'length_hint', 'getitem',\
    'setitem', 'delitem', 'missing', 'iter', 'reversed', 'contains', 'add',\
    'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'div', 'mod', 'divmod',\
    'pow', 'lshift', 'rshift', 'and', 'or', 'xor', 'radd', 'rsub', 'rmul',\
    'rmatmul', 'rtruediv', 'rfloordiv', 'rdiv', 'rmod', 'rdivmod', 'rpow',\
    'rlshift', 'rrshift', 'rand', 'ror', 'rxor', 'iadd', 'isub', 'imul',\
    'imatmul', 'itruediv', 'ifloordiv', 'idiv', 'imod', 'ipow', 'ilshift',\
    'irshift', 'iand', 'ior', 'ixor', 'neg', 'pos', 'abs', 'invert',\
    'complex', 'int', 'float', 'index', 'round', 'trunc', 'floor', 'ceil',\
    'next', 'enter', 'exit', 'await', 'aiter', 'anext', 'aenter', 'aexit']
__pdoc__ = {}
for magic_name in magic_names:
    for class_name in class_names:
        __pdoc__['{0!s}.__{1!s}__'.format(class_name, magic_name)] = True

