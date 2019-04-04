"""
File: distpy/transform/InvertTransform.py
Author: Keith Tauscher
Date: 3 Apr 2019

Description: File containing a function which computes the inverse of
             transforms.
"""
from __future__ import division
from .AffineTransform import AffineTransform
from .ArcsinTransform import ArcsinTransform
from .ArsinhTransform import ArsinhTransform
from .BoxCoxTransform import BoxCoxTransform
from .CompositeTransform import CompositeTransform
from .Exp10Transform import Exp10Transform
from .ExponentialTransform import ExponentialTransform
from .ExponentiatedTransform import ExponentiatedTransform
from .Log10Transform import Log10Transform
from .LoggedTransform import LoggedTransform
from .LogTransform import LogTransform
from .LogisticTransform import LogisticTransform
from .NullTransform import NullTransform
from .PowerTransform import PowerTransform
from .ProductTransform import ProductTransform
from .ReciprocalTransform import ReciprocalTransform
from .SumTransform import SumTransform
from .SineTransform import SineTransform

def invert_transform(transform):
    """
    Computes the Transform that is the inverse of the given Transform object.
    
    transform: a Transform object
    
    returns: a Transform object which represents the inverse of transform
    """
    if isinstance(transform, AffineTransform):
        return AffineTransform(1 / transform.scale_factor,\
            ((-1) * transform.translation) / transform.scale_factor)
    elif isinstance(transform, ArcsinTransform):
        return SineTransform()
    elif isinstance(transform, ArsinhTransform):
        return ArsinhTransform((-1) * transform.shape)
    elif isinstance(transform, BoxCoxTransform):
        if transform.power == 0:
            return CompositeTransform(ExponentialTransform(),\
                AffineTransform(1, (-1) * transform.offset))
        else:
            inverse = AffineTransform(transform.power, 1)
            inverse = CompositeTransform(inverse,\
                PowerTransform(1 / transform.power))
            inverse = CompositeTransform(inverse,\
                AffineTransform(1, (-1) * transform.offset))
            return inverse
    elif isinstance(transform, CompositeTransform):
        return CompositeTransform(invert_transform(transform.outer_transform),\
            invert_transform(transform.inner_transform))
    elif isinstance(transform, Exp10Transform):
        return Log10Transform()
    elif isinstance(transform, ExponentialTransform):
        return LogTransform()
    elif isinstance(transform, ExponentiatedTransform):
        return CompositeTransform(LogTransform(),\
            invert_transform(transform.transform))
    elif isinstance(transform, Log10Transform):
        return Exp10Transform()
    elif isinstance(transform, LogTransform):
        return ExponentialTransform()
    elif isinstance(transform, LoggedTransform):
        return CompositeTransform(ExponentialTransform(),\
            invert_transform(transform.transform))
    elif isinstance(transform, LogisticTransform):
        inverse = AffineTransform(-1, 0)
        inverse = CompositeTransform(inverse, ExponentialTransform())
        inverse = CompositeTransform(inverse, AffineTransform(1, 1))
        inverse = CompositeTransform(inverse, ReciprocalTransform())
        return inverse
    elif isinstance(transform, NullTransform):
        return NullTransform()
    elif isinstance(transform, PowerTransform):
        return PowerTransform(1 / transform.power)
    elif isinstance(transform, ReciprocalTransform):
        return ReciprocalTransform()
    elif isinstance(transform, SineTransform):
        return ArcsinTransform()
    elif isinstance(transform, ProductTransform):
        raise NotImplementedError("ProductTransform objects cannot be " +\
            "simply inverted.")
    elif isinstance(transform, SumTransform):
        raise NotImplementedError("SumTransform objects cannot be simply " +\
            "inverted.")
    else:
        raise ValueError("transform was not recognized as a Transform object.")

