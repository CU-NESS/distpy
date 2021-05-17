"""
Module containing function (`distpy.transform.CastTransform.cast_to_transform`)
which casts strings/objects to `distpy.transform.Transform.Transform` objects.
This casting is loose. For example, `None` is cast to a
`distpy.transform.NullTransform.NullTransform` object, `'ln'` is cast to a
`distpy.transform.LogTransform.LogTransform`, and any
`distpy.transform.Transform.Transform` object is guaranteed to cast into
itself. This module also contains a function
(`distpy.transform.CastTransform.castable_to_transform`) which returns a
boolean describing whether a given key object can be cast into a
`distpy.transform.Transform.Transform`.

**File**: $DISTPY/distpy/transform/CastTransform.py  
**Author**: Keith Tauscher  
**Date**: 16 May 2021
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
    Loads a `distpy.transform.Transform.Transform` from the given `key`.
    
    Parameters
    ----------
    key : str or None or `distpy.transform.Transform.Transform`
        object to cast to a `distpy.transform.Transform.Transform`
    
    Returns
    -------
    casted_transform : `distpy.transform.Transform.Transform`
        object cast from `key`
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
            elif lower_cased_key == ['sin', 'sine']:
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
    Function determining whether the given key can be cast into a
    `distpy.transform.Transform.Transform` using the
    `distpy.transform.CastTransform.cast_to_transform` function.
    
    Parameters
    ----------
    key : object
        object to check for castability. See
        `distpy.transform.CastTransform.cast_to_transform` function for what
        types of `key` will work
    return_transform_if_true : bool
        determines what should be returned if `key` can be successfully
        cast to a `distpy.transform.Transform.Transform`
    
    Returns
    -------
    cast_result : bool or `distpy.transform.Transform.Transform`
        - if `key` can be successfully cast to a
        `distpy.transform.Transform.Transform`, this method returns:
            - the casted `distpy.transform.Transform.Transform` if
            `return_transform_if_true` is True
            - True if `return_transform_if_true` is False
        - if `key` cannot be successfully cast to a
        `distpy.transform.Transform.Transform`, this method returns False
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

