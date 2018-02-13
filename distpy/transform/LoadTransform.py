"""
File: distpy/transform/LoadTransform.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing functions which load Transform objects from hdf5
             files or hdf5 file groups.
"""
from .NullTransform import NullTransform
from .LogTransform import LogTransform
from .ExponentialTransform import ExponentialTransform
from .Log10Transform import Log10Transform
from .SquareTransform import SquareTransform
from .ArcsinTransform import ArcsinTransform
from .LogisticTransform import LogisticTransform
from .AffineTransform import AffineTransform
from .ReciprocalTransform import ReciprocalTransform
from .ExponentiatedTransform import ExponentiatedTransform
from .LoggedTransform import LoggedTransform
from .SumTransform import SumTransform
from .ProductTransform import ProductTransform
from .CompositeTransform import CompositeTransform

try:
    import h5py
except:
    have_h5py = False
    no_h5py_error = NotImplementedError("Loading couldn't be completed " +\
        "because h5py couldn't be imported.")
else:
    have_h5py = True

def load_transform_from_hdf5_group(group):
    """
    Loads a Transform object from an hdf5 file group.
    
    group: the hdf5 file group from which to load the Transform
    
    returns: Transform object of the correct type
    """
    try:
        class_name = group.attrs['class']
    except KeyError:
        raise ValueError("group does not appear to contain a transform.")
    if class_name == 'NullTransform':
        return NullTransform()
    elif class_name == 'AffineTransform':
        scale_factor = group.attrs['scale_factor']
        translation = group.attrs['translation']
        return AffineTransform(scale_factor, translation)
    elif class_name == 'LogTransform':
        return LogTransform()
    elif class_name == 'Log10Transform':
        return Log10Transform()
    elif class_name == 'SquareTransform':
        return SquareTransform()
    elif class_name == 'ArcsinTransform':
        return ArcsinTransform()
    elif class_name == 'LogisticTransform':
        return LogisticTransform()
    elif class_name == 'ExponentialTransform':
        return ExponentialTransform()
    elif class_name == 'ReciprocalTransform':
        transform = load_transform_from_hdf5_group(group['transform'])
        return ReciprocalTransform(transform)
    elif class_name == 'ExponentiatedTransform':
        transform = load_transform_from_hdf5_group(group['transform'])
        return ExponentiatedTransform(transform)
    elif class_name == 'LoggedTransform':
        transform = load_transform_from_hdf5_group(group['transform'])
        return LoggedTransform(transform)
    elif class_name == 'SumTransform':
        transform_0 = load_transform_from_hdf5_group(group['transform_0'])
        transform_1 = load_transform_from_hdf5_group(group['transform_1'])
        return SumTransform(transform_0, transform_1)
    elif class_name == 'ProductTransform':
        transform_0 = load_transform_from_hdf5_group(group['transform_0'])
        transform_1 = load_transform_from_hdf5_group(group['transform_1'])
        return ProductTransform(transform_0, transform_1)
    elif class_name == 'CompositeTransform':
        inner_transform =\
            load_transform_from_hdf5_group(group['inner_transform'])
        outer_transform =\
            load_transform_from_hdf5_group(group['outer_transform'])
        return CompositeTransform(inner_transform, outer_transform)
    else:
        raise ValueError("class of transform not recognized.")

def load_transform_from_hdf5_file(file_name):
    """
    Loads a Transform object from the given hdf5 file.
    
    file_name: string name of hdf5 file from which to load Transform object
    
    return: Transform object of the correct type
    """
    if have_h5py:
        hdf5_file = h5py.File(file_name, 'r')
        transform = load_transform_from_hdf5_group(hdf5_file)
        hdf5_file.close()
        return transform
    else:
        raise no_h5py_error

