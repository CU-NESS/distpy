"""
File: distpy/transform/CastTransform.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: File containing function (cast_to_transform) which cast
             strings/objects to Transform objects. This casting is loose. For
             example, None is cast to a NullTransform object, 'ln' is cast to a
             LogTransform, and any Transform object is guaranteed to cast into
             itself. This file also contains a function (castable_to_transform)
             which returns a boolean describing whether a given key object can
             be cast into a Transform.
"""
from .Transform import Transform
from .NullTransform import NullTransform
from .LogTransform import LogTransform
from .Log10Transform import Log10Transform
from .SquareTransform import SquareTransform
from .ArcsinTransform import ArcsinTransform
from .LogisticTransform import LogisticTransform

def cast_to_transform(key):
    """
    Loads a Transform from the given string key.
    
    key: either (1) None, (2) a string key from specifying which transform to
         load, or (3) a Transform object which will be parroted back
    
    returns: Transform object of the correct type
    """
    if key is None:
        return NullTransform()
    elif isinstance(key, str):
        lower_cased_key = key.lower()
        if lower_cased_key in ['null', 'none']:
            return NullTransform()
        elif lower_cased_key in ['log', 'ln']:
            return LogTransform()
        elif lower_cased_key == 'log10':
            return Log10Transform()
        elif lower_cased_key == 'square':
            return SquareTransform()
        elif lower_cased_key == 'arcsin':
            return ArcsinTransform()
        elif lower_cased_key == 'logistic':
            return LogisticTransform()
        else:
            raise ValueError("transform could not be reconstructed from " +\
                             "key, " + key + ", as key was not understood.")
    elif isinstance(key, Transform):
        return key
    else:
        raise TypeError("key cannot be cast to transform because it is " +\
                        "neither None nor a string nor a Transform.")

def castable_to_transform(key, return_transform_if_true=False):
    """
    Function determining whether the given key can be cast into a Transform
    object.
    
    key: either (1) None, (2) a string key from specifying which transform to
         load, or (3) a Transform object which will be parroted back
    return_transform_if_true: If True and the given key can successfully be
                              cast to a Transform object, that actual Transform
                              object is returned. Otherwise, this parameter has
                              no effect. If False (default), this function is
                              guaranteed to return a bool.
    
    returns: False: if key cannot be cast into a Transform without an error
             True: if key can be cast into a Transform without an error and
                   return_transform_if_true is False
             a Transform object: if key can be cast into a Transform without an
                                 error and return_transform_if_true is True
    """
    try:
        transform = cast_to_transform(key)
    except:
        return False
    else:
        if return_transform_if_true:
            return transform
        else:
            return True

