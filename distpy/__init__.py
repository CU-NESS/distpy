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
    confidence_contour_2D, bivariate_histogram, triangle_plot, Expression
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

class_names = ['distpy.util.Expression.Expression',\
    'distpy.transform.Transform.Transform',\
    'distpy.transform.TransformSet.TransformSet',\
    'distpy.transform.TransformList.TransformList',\
    'distpy.transform.AffineTransform.AffineTransform',\
    'distpy.transform.ArcsinTransform.ArcsinTransform',\
    'distpy.transform.ArsinhTransform.ArsinhTransform',\
    'distpy.transform.BoxCoxTransform.BoxCoxTransform',\
    'distpy.transform.CompositeTransform.CompositeTransform',\
    'distpy.transform.Exp10Transform.Exp10Transform',\
    'distpy.transform.ExponentialTransform.ExponentialTransform',\
    'distpy.transform.ExponentiatedTransform.ExponentiatedTransform',\
    'distpy.transform.LogTransform.LogTransform',\
    'distpy.transform.Log10Transform.Log10Transform',\
    'distpy.transform.LoggedTransform.LoggedTransform',\
    'distpy.transform.LogisticTransform.LogisticTransform',\
    'distpy.transform.NullTransform.NullTransform',\
    'distpy.transform.PowerTransform.PowerTransform',\
    'distpy.transform.ProductTransform.ProductTransform',\
    'distpy.transform.ReciprocalTransform.ReciprocalTransform',\
    'distpy.transform.SineTransform.SineTransform',\
    'distpy.transform.SumTransform.SumTransform']
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

