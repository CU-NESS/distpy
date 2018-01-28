"""
File: distpy/transform/LoadTransformList.py
Author: Keith Tauscher
Date: 27 Jan 2018

Description: File containing function which can load TransformList objects from
             an hdf5 file group.
"""
from .LoadTransform import load_transform_from_hdf5_group
from .TransformList import TransformList

try:
    import h5py
except:
    have_h5py = False
    no_h5py_error = NotImplementedError("Loading couldn't be completed " +\
        "because h5py couldn't be imported.")
else:
    have_h5py = True

def load_transform_list_from_hdf5_group(group):
    """
    Loads a TransformList from an hdf5 file group in which it was saved.
    
    group: the hdf5 file group from which to load a TransformList object
    
    returns: TransformList object contained in the hdf5 file group
    """
    transforms = []
    while 'transform_{}'.format(len(transforms)) in group:
        subgroup = group['transform_{}'.format(len(transforms))]
        transforms.append(load_transform_from_hdf5_group(subgroup))
    return TransformList(*transforms)

def load_transform_list_from_hdf5_file(file_name):
    """
    Loads a TransformList from an hdf5 file in which it was saved.
    
    file_name: location of hdf5 file containing data for this TransformList
    
    returns: TransformList object contained in the hdf5 file
    """
    if have_h5py:
        hdf5_file = h5py.File(file_name, 'r')
        transform_list = load_transform_list_from_hdf5_group(hdf5_file)
        hdf5_file.close()
        return transform_list
    else:
        raise no_h5py_error

