"""
Module containing functions which load `distpy.transform.Transform.Transform`
objects from hdf5 files or hdf5 file groups.

**File**: $DISTPY/distpy/transform/LoadTransform.py  
**Author**: Keith Tauscher  
**Date**: 16 May 2021
"""
from .NullTransform import NullTransform
from .BoxCoxTransform import BoxCoxTransform
from .LogTransform import LogTransform
from .ArsinhTransform import ArsinhTransform
from .ExponentialTransform import ExponentialTransform
from .Exp10Transform import Exp10Transform
from .Log10Transform import Log10Transform
from .PowerTransform import PowerTransform
from .SineTransform import SineTransform
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
    Loads a `distpy.transform.Transform.Transform` object from an hdf5 file
    group.
    
    Parameters
    ----------
    group : h5py.Group
        the hdf5 file group from which to load the
        `distpy.transform.Transform.Transform`
    
    Returns
    -------
    loaded_transform : `distpy.transform.Transform.Transform`
        object loaded from the given group
    """
    try:
        class_name = group.attrs['class']
    except KeyError:
        raise ValueError("group does not appear to contain a transform.")
    if class_name == 'NullTransform':
        return NullTransform()
    elif class_name == 'BoxCoxTransform':
        power = group.attrs['power']
        offset = group.attrs['offset']
        return BoxCoxTransform(power, offset=offset)
    elif class_name == 'ArsinhTransform':
        shape = group.attrs['shape']
        return ArsinhTransform(shape)
    elif class_name == 'AffineTransform':
        scale_factor = group.attrs['scale_factor']
        translation = group.attrs['translation']
        return AffineTransform(scale_factor, translation)
    elif class_name == 'LogTransform':
        return LogTransform()
    elif class_name == 'Log10Transform':
        return Log10Transform()
    elif class_name == 'SineTransform':
        return SineTransform()
    elif class_name == 'ArcsinTransform':
        return ArcsinTransform()
    elif class_name == 'LogisticTransform':
        return LogisticTransform()
    elif class_name == 'ExponentialTransform':
        return ExponentialTransform()
    elif class_name == 'Exp10Transform':
        return Exp10Transform()
    elif class_name == 'PowerTransform':
        return PowerTransform(group.attrs['power'])
    elif class_name == 'ReciprocalTransform':
        return ReciprocalTransform()
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
    Loads a `distpy.transform.Transform.Transform` object from an hdf5 file.
    
    Parameters
    ----------
    file_name : str
        the name of the hdf5 file from which to load the
        `distpy.transform.Transform.Transform`
    
    Returns
    -------
    loaded_transform : `distpy.transform.Transform.Transform`
        object loaded from the given file
    """
    if have_h5py:
        hdf5_file = h5py.File(file_name, 'r')
        transform = load_transform_from_hdf5_group(hdf5_file)
        hdf5_file.close()
        return transform
    else:
        raise no_h5py_error

