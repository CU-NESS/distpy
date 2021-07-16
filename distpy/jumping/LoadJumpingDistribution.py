"""
Module containing functions that load
`distpy.jumping.JumpingDistribution.JumpingDistribution` objects from hdf5
groups and hdf5 files.

**File**: $DISTPY/distpy/jumping/LoadJumpingDistribution.py  
**Author**: Keith Tauscher  
**Date**: 3 Jul 2021
"""
from ..util import get_hdf5_value
from ..distribution import load_distribution_from_hdf5_group
from .GaussianJumpingDistribution import GaussianJumpingDistribution
from .SourceDependentGaussianJumpingDistribution\
    import SourceDependentGaussianJumpingDistribution
from .TruncatedGaussianJumpingDistribution\
    import TruncatedGaussianJumpingDistribution
from .UniformJumpingDistribution import UniformJumpingDistribution
from .BinomialJumpingDistribution import BinomialJumpingDistribution
from .AdjacencyJumpingDistribution import AdjacencyJumpingDistribution
from .GridHopJumpingDistribution import GridHopJumpingDistribution
from .SourceIndependentJumpingDistribution\
    import SourceIndependentJumpingDistribution
from .LocaleIndependentJumpingDistribution\
    import LocaleIndependentJumpingDistribution
from .JumpingDistributionSum import JumpingDistributionSum

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
    Loads a `distpy.jumping.JumpingDistribution.JumpingDistribution` from the
    given hdf5 group.
    
    Parameters
    ----------
    group : h5py.Group
        the hdf5 file group from which to load the
        `distpy.jumping.JumpingDistribution.JumpingDistribution`
    
    Returns
    -------
    loaded : `distpy.jumping.JumpingDistribution.JumpingDistribution`
        distribution object of the correct type
    """
    try:
        class_name = group.attrs['class']
    except KeyError:
        raise ValueError("group given does not appear to contain a jumping " +\
            "distribution.")
    if class_name == 'JumpingDistributionSum':
        subgroup = group['jumping_distributions']
        idistribution = 0
        inner_class_names = []
        while '{:d}'.format(idistribution) in subgroup:
            inner_class_names.append(\
                subgroup['{:d}'.format(idistribution)].attrs['class'])
            idistribution += 1
        args = [eval(inner_class_name)\
                for inner_class_name in inner_class_names]
    else:
        args = []
    try:
        cls = eval(class_name)
    except:
        raise ValueError("The class of the Distribution was not recognized.")
    return cls.load_from_hdf5_group(group, *args)

def load_jumping_distribution_from_hdf5_file(file_name):
    """
    Loads a `distpy.jumping.JumpingDistribution.JumpingDistribution` from the
    given hdf5 file.
    
    Parameters
    ----------
    file_name : str
        name of the hdf5 file from which to load the
        `distpy.jumping.JumpingDistribution.JumpingDistribution`
    
    Returns
    -------
    loaded : `distpy.jumping.JumpingDistribution.JumpingDistribution`
        distribution object of the correct type
    """
    if have_h5py:
        hdf5_file = h5py.File(file_name, 'r')
        jumping_distribution =\
            load_jumping_distribution_from_hdf5_group(hdf5_file)
        hdf5_file.close()
        return jumping_distribution
    else:
        raise no_h5py_error

