"""
File: distpy/jumping/LoadJumpingDistributionSet.py
Author: Keith Tauscher
Date: 22 Dec 2017

Description: File containing functions which load JumpingDistributionSet
             objects from hdf5 groups or hdf5 files.
"""
from ..transform import load_transform_from_hdf5_group
from .JumpingDistributionSet import JumpingDistributionSet
from .LoadJumpingDistribution import load_jumping_distribution_from_hdf5_group

try:
    import h5py
except:
    have_h5py = False
    no_h5py_error = NotImplementedError("Loading couldn't be completed " +\
        "because h5py couldn't be imported.")
else:
    have_h5py = True

def load_jumping_distribution_set_from_hdf5_group(group):
    """
    Loads JumpingDistributionSet object from the given hdf5 group.
    
    group: hdf5 file group from which to read data about the
           JumpingDistributionSet
    
    returns: JumpingDistributionSet object
    """
    ituple = 0
    jumping_distribution_tuples = []
    while ('distribution_{}'.format(ituple)) in group:
        subgroup = group['distribution_{}'.format(ituple)]
        distribution = load_jumping_distribution_from_hdf5_group(subgroup)
        params = []
        transforms = []
        iparam = 0
        for iparam in range(distribution.numparams):
            params.append(subgroup.attrs['parameter_{}'.format(iparam)])
            subsubgroup = subgroup['transform_{}'.format(iparam)]
            transforms.append(load_transform_from_hdf5_group(subsubgroup))
        jumping_distribution_tuples.append((distribution, params, transforms))
        ituple += 1
    return JumpingDistributionSet(\
        jumping_distribution_tuples=jumping_distribution_tuples)

def load_jumping_distribution_set_from_hdf5_file(file_name):
    """
    Loads a JumpingDistributionSet from an hdf5 file in which it was saved.
    
    file_name: location of hdf5 file containing data for JumpingDistributionSet
    
    returns: JumpingDistributionSet object contained in the hdf5 file
    """
    if have_h5py:
        hdf5_file = h5py.File(file_name, 'r')
        jumping_distribution_set =\
            load_jumping_distribution_set_from_hdf5_group(hdf5_file)
        hdf5_file.close()
        return jumping_distribution_set
    else:
        raise no_h5py_error

