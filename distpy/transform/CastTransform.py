"""
File: distpy/transform/CastTransform.py
Author: Keith Tauscher
Date: 12 Feb 2018

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
from .ReciprocalTransform import ReciprocalTransform
from .AffineTransform import AffineTransform
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

def cast_to_transform(key):
    """
    Loads a Transform from the given string key.
    
    key: either (1) None, (2) a string key from specifying which transform to
         load, or (3) a Transform object which will be parroted back
    
    returns: Transform object of the correct type
    """
    if type(key) is type(None):
        return NullTransform()
    elif isinstance(key, basestring):
        key_not_understood_error = ValueError(("transform could not be " +\
            "reconstructed from key, {!s}, as key was not " +\
            "understood.").format(key))
        lower_cased_key = key.lower()
        split_lower_cased_key = lower_cased_key.split(' ')
        num_tokens = len(split_lower_cased_key)
        if num_tokens == 1:
            if lower_cased_key in ['null', 'none']:
                return NullTransform()
            elif lower_cased_key in ['log', 'ln']:
                return LogTransform()
            elif lower_cased_key == 'log10':
                return Log10Transform()
            elif lower_cased_key == 'sine':
                return SineTransform()
            elif lower_cased_key == 'arcsin':
                return ArcsinTransform()
            elif lower_cased_key == 'logistic':
                return LogisticTransform()
            elif lower_cased_key == 'exp':
                return ExponentialTransform()
            elif lower_cased_key == 'exp10':
                return Exp10Transform()
            elif lower_cased_key == 'reciprocal':
                return ReciprocalTransform()
            else:
                raise key_not_understood_error
        elif num_tokens == 2:
            if split_lower_cased_key[0] == 'scale':
                return AffineTransform(float(split_lower_cased_key[1]), 0)
            elif split_lower_cased_key[0] == 'translate':
                return AffineTransform(1, float(split_lower_cased_key[1]))
            elif split_lower_cased_key[0] in ['boxcox', 'box-cox']:
                return\
                    BoxCoxTransform(float(split_lower_cased_key[1]), offset=0)
            elif split_lower_cased_key[0] == 'arsinh':
                return ArsinhTransform(float(split_lower_cased_key[1]))
            elif split_lower_cased_key[0] == 'power':
                return PowerTransform(float(split_lower_cased_key[1]))
            else:
                raise key_not_understood_error
        elif num_tokens == 3:
            if split_lower_cased_key[0] == 'affine':
                scale_factor = float(split_lower_cased_key[1])
                translation = float(split_lower_cased_key[2])
                return AffineTransform(scale_factor, translation)
            elif split_lower_cased_key[0] in ['boxcox', 'box-cox']:
                return BoxCoxTransform(float(split_lower_cased_key[1]),\
                    offset=float(split_lower_cased_key[2]))
            else:
                raise key_not_understood_error
        else:
            raise key_not_understood_error
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

