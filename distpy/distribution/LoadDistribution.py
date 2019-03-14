"""
File: distpy/distribution/LoadDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing functions which can load any Distribution (and it
             determines which needs to be loaded!) from an hdf5 file group or
             file in which it was saved.
"""
from ..util import get_hdf5_value
from .WindowedDistribution import WindowedDistribution
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
from .UniformConditionDistribution import UniformConditionDistribution
from .EllipticalUniformDistribution import EllipticalUniformDistribution
from .TruncatedGaussianDistribution import TruncatedGaussianDistribution
from .SechDistribution import SechDistribution
from .SechSquaredDistribution import SechSquaredDistribution
from .GaussianDistribution import GaussianDistribution
from .ParallelepipedDistribution import ParallelepipedDistribution
from .LinkedDistribution import LinkedDistribution
from .SequentialDistribution import SequentialDistribution
from .GriddedDistribution import GriddedDistribution
from .UniformDirectionDistribution import UniformDirectionDistribution
from .GaussianDirectionDistribution import GaussianDirectionDistribution
from .UniformTriangulationDistribution import UniformTriangulationDistribution
from .CustomDiscreteDistribution import CustomDiscreteDistribution
from .DiscreteUniformDistribution import DiscreteUniformDistribution
from .DistributionSum import DistributionSum
from .DeterministicDistribution import DeterministicDistribution
from .DistributionList import DistributionList

try:
    import h5py
except:
    have_h5py = False
    no_h5py_error = NotImplementedError("Loading couldn't be completed " +\
        "because h5py couldn't be imported.")
else:
    have_h5py = True

def load_distribution_from_hdf5_group(group, *args):
    """
    Loads a distribution from the given hdf5 group.
    
    group: the hdf5 file group from which to load the distribution
    
    returns: Distribution object of the correct type
    """
    try:
        class_name = group.attrs['class']
        if class_name == 'DistributionSum':
            subgroup = group['distributions']
            idistribution = 0
            inner_class_names = []
            while '{:d}'.format(idistribution) in subgroup:
                inner_class_names.append(\
                    subgroup['{:d}'.format(idistribution)].attrs['class'])
                idistribution += 1
            args = [eval(inner_class_name)\
                for inner_class_name in inner_class_names] +\
                [arg for arg in args]
        elif class_name in ['LinkedDistribution', 'SequentialDistribution']:
            inner_class_name = group['shared_distribution'].attrs['class']
            args = [eval(inner_class_name)] + [arg for arg in args]
        elif class_name == 'DistributionList':
            idistribution = 0
            inner_class_names = []
            while 'distribution_{:d}'.format(idistribution) in group:
                subgroup = group['distribution_{:d}'.format(idistribution)]
                inner_class_names.append(subgroup.attrs['class'])
                idistribution += 1
            args = [eval(inner_class_name)\
                for inner_class_name in inner_class_names] +\
                [arg for arg in args]
        elif class_name == 'LinearDirectionDistribution':
            angle_distribution_class_name =\
                group['angle_distribution'].attrs['class']
            args =\
                [eval(angle_distribution_class_name)] + [arg for arg in args]
        elif class_name == 'WindowedDistribution':
            background_distribution_class_name =\
                group['background_distribution'].attrs['class']
            foreground_distribution_class_name =\
                group['foreground_distribution'].attrs['class']
            args = [eval(background_distribution_class_name),\
                eval(foreground_distribution_class_name)] +\
                [arg for arg in args]
        else:
            args = [arg for arg in args]
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

