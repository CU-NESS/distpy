"""
File: distpy/jumping/LoadJumpingDistribution.py
Author: Keith Tauscher
Date: 20 Dec 2017

Description: File containing functions which load JumpingDistribution objects
             from hdf5 groups and hdf5 files.
"""
from ..util import get_hdf5_value
from ..distribution import load_distribution_from_hdf5_group
from .GaussianJumpingDistribution import GaussianJumpingDistribution
from .TruncatedGaussianJumpingDistribution\
    import TruncatedGaussianJumpingDistribution
from .UniformJumpingDistribution import UniformJumpingDistribution
from .BinomialJumpingDistribution import BinomialJumpingDistribution
from .AdjacencyJumpingDistribution import AdjacencyJumpingDistribution
from .SourceIndependentJumpingDistribution\
    import SourceIndependentJumpingDistribution
from .LocaleIndependentJumpingDistribution\
    import LocaleIndependentJumpingDistribution

try:
    import h5py
except:
    have_h5py = False
    no_h5py_error = NotImplementedError("Loading couldn't be completed " +\
        "because h5py couldn't be imported.")
else:
    have_h5py = True

def load_jumping_distribution_from_hdf5_group(group):
    """
    Loads a jumping distribution from the given hdf5 group.
    
    group: the hdf5 file group from which to load the jumping distribution
    
    returns: JumpingDistribution object of the correct type
    """
    try:
        class_name = group.attrs['class']
    except KeyError:
        raise ValueError("group given does not appear to contain a jumping " +\
            "distribution.")
    if class_name == 'GaussianJumpingDistribution':
        return GaussianJumpingDistribution(get_hdf5_value(group['covariance']))
    elif class_name == 'TruncatedGaussianJumpingDistribution':
        variance = group.attrs['variance']
        if 'low' in group.attrs:
            low = group.attrs['low']
        else:
            low = None
        if 'high' in group.attrs:
            high = group.attrs['high']
        else:
            high = None
        return\
            TruncatedGaussianJumpingDistribution(variance, low=low, high=high)
    elif class_name == 'UniformJumpingDistribution':
        return UniformJumpingDistribution(get_hdf5_value(group['covariance']))
    elif class_name == 'BinomialJumpingDistribution':
        minimum = group.attrs['minimum']
        maximum = group.attrs['maximum']
        return BinomialJumpingDistribution(minimum, maximum)
    elif class_name == 'AdjacencyJumpingDistribution':
        jumping_probability = group.attrs['jumping_probability']
        if 'minimum' in group.attrs:
            minimum = group.attrs['minimum']
        else:
            minimum = None
        if 'maximum' in group.attrs:
            maximum = group.attrs['maximum']
        else:
            maximum = None
        return\
            AdjacencyJumpingDistribution(jumping_probability, minimum, maximum)
    elif class_name == 'LocaleIndependentJumpingDistribution':
        distribution = load_distribution_from_hdf5_group(group['distribution'])
        return LocaleIndependentJumpingDistribution(distribution)
    elif class_name == 'SourceIndependentJumpingDistribution':
        distribution = load_distribution_from_hdf5_group(group['distribution'])
        return SourceIndependentJumpingDistribution(distribution)
    else:
        raise ValueError("The class of the Distribution was not recognized.")

def load_jumping_distribution_from_hdf5_file(file_name):
    """
    Loads JumpingDistribution object of any subclass from an hdf5 file at the
    given file name.
    
    file_name: location of hdf5 file containing jumping distribution
    
    returns: JumpingDistribution object contained in the hdf5 file
    """
    if have_h5py:
        hdf5_file = h5py.File(file_name, 'r')
        jumping_distribution =\
            load_jumping_distribution_from_hdf5_group(hdf5_file)
        hdf5_file.close()
        return jumping_distribution
    else:
        raise no_h5py_error

