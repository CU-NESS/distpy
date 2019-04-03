"""
File: distpy/transform/CastTransformList.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing functions which allow for objects to be cast to
             TransformList objects.
"""
from ..util import sequence_types
from .CastTransform import castable_to_transform, cast_to_transform
from .TransformList import TransformList

def cast_to_transform_list(key, num_transforms=None):
    """
    Casts key into a TransformList object. If num_transforms is non-None, this
    function can also cast to a TransformList object of a specific length.
    """
    if isinstance(key, TransformList):
        if (type(num_transforms) is not type(None)) and\
            (len(key) != num_transforms):
            raise ValueError("The given TransformList was not of the " +\
                "specified length. So, it could not be cast successfully " +\
                "into a TransformList of the desired size.")
        else:
            return key
    elif type(key) in sequence_types:
        if (type(num_transforms) is not type(None)) and\
            (len(key) != num_transforms):
            raise ValueError("The given sequence was not of the specified " +\
                "length. So, it could not be cast successfully into a " +\
                "TransformList of the desired size.")
        else:
            return TransformList(*key)
    elif castable_to_transform(key):
        transform = cast_to_transform(key)
        if type(num_transforms) is type(None):
            return TransformList(transform)
        else:
            return TransformList(*([transform] * num_transforms))
    else:
        raise TypeError("key could not be cast to a TransformList object.")

def castable_to_transform_list(key, return_transform_list_if_true=False,\
    num_transforms=None):
    """
    Function determining whether the given key can be cast into a TransformList
    object.
    
    key: either (1) a TransformList object, (2) a 
    return_transform_list_if_true: If True and the given key can successfully
                                   be cast to a TransformList object, that
                                   actual TransformList object is returned.
                                   Otherwise, this parameter has no effect. If
                                   False (default), this function is guaranteed
                                   to return a bool.
    num_transforms: if None, this function checks whether key can be cast into
                             any TransformList.
                    if an int, this function checks whether key can be cast
                               a TransformList with this number transforms
    
    returns: False: if key cannot be cast into a Transform without an error
             True: if key can be cast into a Transform without an error and
                   return_transform_if_true is False
             a Transform object: if key can be cast into a Transform without an
                                 error and return_transform_if_true is True
    """
    try:
        transform_list =\
            cast_to_transform_list(key, num_transforms=num_transforms)
    except:
        return False
    else:
        if return_transform_list_if_true:
            return transform_list
        else:
            return True

