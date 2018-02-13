"""
File: distpy/jumping/LocaleIndependentJumpingDistribution.py
Author: Keith Tauscher
Date: 21 Dec 2017

Description: File containing a class which represents a jumping distribution
             whose value is dependent only on the displacement between the
             source and destination points.
"""
import numpy as np
from ..util import int_types
from ..distribution import Distribution, load_distribution_from_hdf5_group
from .JumpingDistribution import JumpingDistribution

class LocaleIndependentJumpingDistribution(JumpingDistribution):
    """
    Class which represents a degenerate sort of jumping distribution: one whose
    value is independent of the source of the jump.
    """
    def __init__(self, distribution):
        """
        Initializes this LocaleIndependentJumpingDistribution with the given
        core distribution.
        
        distribution: a Distribution object describing the probability of the
                      displacement between the source and destination.
        """
        self.distribution = distribution
    
    @property
    def distribution(self):
        """
        Property storing the distribution describing the probability of the
        displacement between the source and destination.
        """
        if not hasattr(self, '_distribution'):
            raise AttributeError("distribution referenced before it was set.")
        return self._distribution
    
    @distribution.setter
    def distribution(self, value):
        """
        Setter for the core distribution of this
        LocaleIndependentJumpingDistribution.
        
        value: a Distribution object describing the probability of the
               displacement between the source and destination
        """
        if isinstance(value, Distribution):
            self._distribution = value
        else:
            raise TypeError("distribution was not a Distribution object.")
    
    def draw(self, source, shape=None):
        """
        Draws a destination point from this jumping distribution given a source
        point.
        
        source: if this JumpingDistribution is univariate, source should be a
                                                           single number
                otherwise, source should be numpy.ndarray of shape (numparams,)
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        
        returns: either single value (if distribution is 1D and shape is None)
                 or array of values
        """
        none_shape = (shape is None)
        if none_shape:
            return source + self.distribution.draw()
        elif type(shape) in int_types:
            shape = (shape,)
        if self.numparams == 1:
            return source + self.distribution.draw(shape=shape)
        else:
            return source[((np.newaxis,) * len(shape)) + (slice(None),)] +\
                self.distribution.draw(shape=shape)
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF: ln(f(source->destination))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        return self.distribution.log_value(destination - source)
    
    def log_value_difference(self, source, destination):
        """
        Computes the log-PDF difference:
        ln(f(source->destination)/f(destination->source))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number difference between one-way log-PDF's
        """
        displacement = destination - source
        return self.distribution.log_value(displacement) -\
            self.distribution.log_value(-displacement)
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. It must be implemented by all subclasses.
        """
        return self.distribution.numparams
    
    def __eq__(self, other):
        """
        Tests for equality between this jumping distribution and other. All
        subclasses must implement this function.
        
        other: JumpingDistribution with which to check for equality
        
        returns: True or False
        """
        if isinstance(other, LocaleIndependentJumpingDistribution):
            return (self.distribution == other.distribution)
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this jumping
        distribution.
        
        group: hdf5 file group to fill with information about this jumping
               distribution
        """
        group.attrs['class'] = 'LocaleIndependentJumpingDistribution'
        self.distribution.fill_hdf5_group(group.create_group('distribution'))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a LocaleIndependentJumpingDistribution from the given hdf5 file
        group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this LocaleIndependentJumpingDistribution was saved
        
        returns: a LocaleIndependentJumpingDistribution object created from the
                 information in the given group
        """
        try:
            assert\
                group.attrs['class'] == 'LocaleIndependentJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "LocaleIndependentJumpingDistribution.")
        distribution = load_distribution_from_hdf5_group(group['distribution'])
        return LocaleIndependentJumpingDistribution(distribution)

