"""
File: distpy/distribution/LoadDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing functions which can load any Distribution (and it
             determines which needs to be loaded!) from an hdf5 file group or
             file in which it was saved.
"""
from ..util import get_hdf5_value
from .UniformDistribution import UniformDistribution
from .GeneralizedParetoDistribution import GeneralizedParetoDistribution
from .GammaDistribution import GammaDistribution
from .ChiSquaredDistribution import ChiSquaredDistribution
from .BetaDistribution import BetaDistribution
from .PoissonDistribution import PoissonDistribution
from .GeometricDistribution import GeometricDistribution
from .BinomialDistribution import BinomialDistribution
from .ExponentialDistribution import ExponentialDistribution
from .DoubleSidedExponentialDistribution import\
    DoubleSidedExponentialDistribution
from .WeibullDistribution import WeibullDistribution
from .KroneckerDeltaDistribution import KroneckerDeltaDistribution
from .InfiniteUniformDistribution import InfiniteUniformDistribution
from .EllipticalUniformDistribution import EllipticalUniformDistribution
from .TruncatedGaussianDistribution import TruncatedGaussianDistribution
from .GaussianDistribution import GaussianDistribution
from .ParallelepipedDistribution import ParallelepipedDistribution
from .LinkedDistribution import LinkedDistribution
from .SequentialDistribution import SequentialDistribution
from .GriddedDistribution import GriddedDistribution
from .UniformDirectionDistribution import UniformDirectionDistribution
from .GaussianDirectionDistribution import GaussianDirectionDistribution
from .UniformTriangulationDistribution import UniformTriangulationDistribution
from .CustomDiscreteDistribution import CustomDiscreteDistribution

try:
    import h5py
except:
    have_h5py = False
    no_h5py_error = NotImplementedError("Loading couldn't be completed " +\
        "because h5py couldn't be imported.")
else:
    have_h5py = True

def load_distribution_from_hdf5_group(group):
    """
    Loads a distribution from the given hdf5 group.
    
    group: the hdf5 file group from which to load the distribution
    
    returns: Distribution object of the correct type
    """
    try:
        class_name = group.attrs['class']
        if class_name in ['LinkedDistribution', 'SequentialDistribution']:
            inner_class_name = group['shared_distribution'].attrs['class']
            args = [eval(inner_class_name)]
        else:
            args = []
        cls = eval(class_name)
    except KeyError:
        raise ValueError("This group doesn't appear to contain a " +\
            "Distribution.")
    return cls.load_from_hdf5_group(group, *args)

def load_distribution_from_hdf5_file(file_name):
    """
    Loads Distribution object of any subclass from an hdf5 file at the given
    file name.
    
    file_name: location of hdf5 file containing distribution
    
    returns: Distribution object contained in the hdf5 file
    """
    if have_h5py:
        hdf5_file = h5py.File(file_name, 'r')
        distribution = load_distribution_from_hdf5_group(hdf5_file)
        hdf5_file.close()
        return distribution
    else:
        raise no_h5py_error

