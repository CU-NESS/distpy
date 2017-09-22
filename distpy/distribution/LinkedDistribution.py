"""
File: distpy/LinkedDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing distribution of many random
             variates which must all be equal.
"""
import numpy as np
from ..util import int_types, numerical_types, sequence_types
from .Distribution import Distribution

class LinkedDistribution(Distribution):
    """
    Class representing a distribution which is shared by an arbitrary number of
    parameters. It piggybacks on another (univariate) distribution (called the
    "shared_distribution") by drawing from it and evaluating its log_value
    while ensuring that the variables linked by this distribution must be
    identical.
    """
    def __init__(self, shared_distribution, numpars):
        """
        Initializes a new LinkedDistribution with the given
        shared_distribution and number of parameters.
        
        shared_distribution: the Distribution which describes how the
                             individual values are distributed (must be a
                             Distribution)
        numpars the number of parameters which this distribution describes
        """
        if isinstance(shared_distribution, Distribution):
            if shared_distribution.numparams == 1:
                self.shared_distribution = shared_distribution
            else:
                raise NotImplementedError("The shared_distribution " +\
                    "provided to a LinkedDistribution was multivariate (I " +\
                    "don't know how to deal with this).")
        else:
            raise ValueError("The shared_distribution given to a " +\
                "LinkedDistribution was not recognizable as a distribution.")
        if (type(numpars) in numerical_types):
            if numpars > 1:
                self._numparams = numpars
            else:
                raise ValueError("A LinkedDistribution was initialized " +\
                    "with only one parameter. Is this really what you want?")
        else:
            raise ValueError("The type of the number of parameters given " +\
                "given to a LinkedDistribution was not numerical.")

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters which this distribution
        describes.
        """
        return self._numparams

    def draw(self, shape=None):
        """
        Draws value from shared_distribution and assigns that value to all
        parameters.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        
        returns numpy.ndarray of values (all are equal by design)
        """
        if shape is None:
            return np.ones(self.numparams) * self.shared_distribution.draw()
        else:
            if type(shape) in int_types:
                shape = (shape,)
            return np.ones(shape + (self.numparams,)) *\
                self.shared_distribution.draw(shape=shape)[...,np.newaxis]

    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution at the
        given point.
        
        point: can be 0D or 1D (if 1D, all values must be identical for
               this function to return something other than -np.inf)
        
        returns: the log of the value of this distribution at the given point
                 (ignoring delta functions)
        """
        if type(point) in numerical_types:
            return self.shared_distribution.log_value(point)
        elif type(point) in sequence_types:
            if (len(point) == self.numparams):
                for ival in range(len(point)):
                    if point[ival] != point[0]:
                        return -np.inf
                return self.shared_distribution.log_value(point[0])
            else:
                raise ValueError("The length of the point given to a " +\
                    "LinkedDistribution was not the same as the " +\
                    "LinkedDistribution's number of parameters.")
        else:
            raise ValueError("The point provided to a LinkedDistribution " +\
                "was not of a numerical type or a list type.")

    def to_string(self):
        """
        Finds and returns a string representation of this LinkedDistribution.
        """
        return "Linked({!s})".format(self.shared_distribution.to_string())
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a LinkedDistribution with the same number of parameters and
        the same shared_distribution and False otherwise.
        """
        if isinstance(other, LinkedDistribution):
            numparams_equal = (self.numparams == other.numparams)
            shared_distribution_equal =\
                (self.shared_distribution == other.shared_distribution)
            return numparams_equal and shared_distribution_equal
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. The
        class name is saved alongside the component distribution and the number
        of parameters.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'LinkedDistribution'
        group.attrs['numparams'] = self.numparams
        subgroup = group.create_group('shared_distribution')
        self.shared_distribution.fill_hdf5_group(subgroup)

