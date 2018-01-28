"""
File: distpy/distribution/LoadDistributionSet.py
Author: Keith Tauscher
"""
from ..transform import load_transform_list_from_hdf5_group
from .DistributionSet import DistributionSet
from .LoadDistribution import load_distribution_from_hdf5_group

try:
    import h5py
except:
    have_h5py = False
    no_h5py_error = NotImplementedError("Loading couldn't be completed " +\
        "because h5py couldn't be imported.")
else:
    have_h5py = True


def load_distribution_set_from_hdf5_group(group):
    """
    Loads DistributionSet object from the given hdf5 group.
    
    group: hdf5 file group from which to read data about the DistributionSet
    
    returns: DistributionSet object
    """
    ituple = 0
    distribution_tuples = []
    while ('distribution_{}'.format(ituple)) in group:
        subgroup = group['distribution_{}'.format(ituple)]
        distribution = load_distribution_from_hdf5_group(subgroup)
        transform_list = load_transform_list_from_hdf5_group(subgroup)
        params = []
        iparam = 0
        for iparam in range(distribution.numparams):
            params.append(subgroup.attrs['parameter_{}'.format(iparam)])
        distribution_tuples.append((distribution, params, transform_list))
        ituple += 1
    return DistributionSet(distribution_tuples=distribution_tuples)

def load_distribution_set_from_hdf5_file(file_name):
    """
    Loads a DistributionSet from an hdf5 file in which it was saved.
    
    file_name: location of hdf5 file containing date for DistributionSet
    
    returns: DistributionSet object contained in the hdf5 file
    """
    if have_h5py:
        hdf5_file = h5py.File(file_name, 'r')
        distribution_set = load_distribution_set_from_hdf5_group(hdf5_file)
        hdf5_file.close()
        return distribution_set
    else:
        raise no_h5py_error

