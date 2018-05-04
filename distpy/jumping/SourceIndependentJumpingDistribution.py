"""
File: distpy/jumping/SourceIndependentJumpingDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing a class which represents a degenerate sort of
             jumping distribution: one whose value is independent of the source
             of the jump.
"""
import numpy as np
from ..distribution import Distribution, load_distribution_from_hdf5_group
from .JumpingDistribution import JumpingDistribution

class SourceIndependentJumpingDistribution(JumpingDistribution):
    """
    Class which represents a degenerate sort of jumping distribution: one whose
    value is independent of the source of the jump.
    """
    def __init__(self, distribution):
        """
        Initializes this SourceIndependentJumpingDistribution with the given
        core distribution.
        
        distribution: a Distribution object describing the probability of
                      jumping to any destination, regardless of source
        """
        self.distribution = distribution
    
    @property
    def distribution(self):
        """
        Property storing the distribution of the destination, regardless of the
        source.
        """
        if not hasattr(self, '_distribution'):
            raise AttributeError("distribution referenced before it was set.")
        return self._distribution
    
    @distribution.setter
    def distribution(self, value):
        """
        Setter for the core distribution of this
        SourceIndependentJumpingDistribution.
        
        value: a Distribution object describing the probability of jumping to
               any destination regardless of source
        """
        if isinstance(value, Distribution):
            self._distribution = value
        else:
            raise TypeError("distribution was not a Distribution object.")
    
    def draw(self, source, shape=None, random=np.random):
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
        return self.distribution.draw(shape=shape, random=random)
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF: ln(f(source->destination))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        return self.distribution.log_value(destination)
    
    def log_value_difference(self, source, destination):
        """
        Computes the log-PDF difference:
        ln(f(source->destination)/f(destination->source))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number difference between one-way log-PDF's
        """
        return self.distribution.log_value(destination) -\
            self.distribution.log_value(source)
    
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
        if isinstance(other, SourceIndependentJumpingDistribution):
            return (self.distribution == other.distribution)
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Property storing boolean describing whether this JumpingDistribution
        describes discrete (True) or continuous (False) variable(s).
        """
        return self.distribution.is_discrete
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this jumping
        distribution.
        
        group: hdf5 file group to fill with information about this jumping
               distribution
        """
        group.attrs['class'] = 'SourceIndependentJumpingDistribution'
        self.distribution.fill_hdf5_group(group.create_group('distribution'))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a SourceIndependentJumpingDistribution from the given hdf5 file
        group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this SourceIndependentJumpingDistribution was saved
        
        returns: a SourceIndependentJumpingDistribution object created from the
                 information in the given group
        """
        try:
            assert\
                group.attrs['class'] == 'SourceIndependentJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "SourceIndependentJumpingDistribution.")
        distribution = load_distribution_from_hdf5_group(group['distribution'])
        return SourceIndependentJumpingDistribution(distribution)

