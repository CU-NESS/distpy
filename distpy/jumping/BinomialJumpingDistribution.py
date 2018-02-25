"""
File: distpy/jumping/BinomialJumpingDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing a jumping distribution on the
             integers where there is a defined minimum and maximum. This one is
             based on the binomial distribution.
"""
import numpy as np
from scipy.special  import gammaln as log_gamma
from ..util import int_types
from .JumpingDistribution import JumpingDistribution

class BinomialJumpingDistribution(JumpingDistribution):
    """
    Class representing a jumping distribution on the integers where there is a
    defined minimum and maximum. This is based on the binomial distribution.
    """
    def __init__(self, minimum, maximum):
        """
        Initializes a BinomialJumpingDistribution with the given extrema.
        
        minimum: minimum allowable value of integer parameter
        maximum: maximum allowable value of integer parameter
        """
        self.minimum = minimum
        self.maximum = maximum
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable integer value of the parameter.
        """
        if not hasattr(self, '_minimum'):
            raise AttributeError("minimum was referenced before it was set.")
        return self._minimum
    
    @minimum.setter
    def minimum(self, value):
        """
        Setter for the minimum allowable integer value of the parameter.
        
        value: an integer
        """
        if type(value) in int_types:
            self._minimum = value
        else:
            raise TypeError("minimum was set to a non-int.")
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable integer value of the parameter.
        """
        if not hasattr(self, '_maximum'):
            raise AttributeError("maximum was referenced before it was set.")
        return self._maximum
    
    @maximum.setter
    def maximum(self, value):
        """
        Setter for the maximum allowable integer value of the parameter.
        
        value: an integer greater than the int stored in the minimum property
        """
        if type(value) in int_types:
            if value > self.minimum:
                self._maximum = value
            else:
                raise ValueError("maximum was set to an int which was less " +\
                    "than or equal to minimum.")
        else:
            raise TypeError("maximum was set to a non-int.")
    
    @property
    def span(self):
        """
        Property storing the difference between the minimum and maximum allowed
        value of the parameter.
        """
        if not hasattr(self, '_span'):
            self._span = (self.maximum - self.minimum)
        return self._span
    
    @property
    def reciprocal_span(self):
        """
        Property storing the reciprocal of the span property.
        """
        if not hasattr(self, '_reciprocal_span'):
            self._reciprocal_span = (1. / self.span)
        return self._reciprocal_span
    
    @property
    def half_reciprocal_span(self):
        """
        Property storing half of the reciprocal of the span property.
        """
        if not hasattr(self, '_half_reciprocal_span'):
            self._half_reciprocal_span = self.reciprocal_span / 2
        return self._half_reciprocal_span
    
    def p_from_shifted_source(self, shifted_source):
        """
        Finds the p value of the binomial distribution associated with the
        given source (which is assumed to already have self.minimum subtracted)
        
        shifted_source: integer between 0 and
                        self.span=self.maximum-self.minimum (inclusive)
        
        returns: p satisfying 0 < p < 1 where pN is near (and usually equal to)
                 shifted_source (p cannot be 0 or 1 because that would imply it
                 could never jump away from the minimum or maximum, and would
                 therefore violate the ergodicity requirement of Markov chains)
        """
        if shifted_source == 0:
            return self.half_reciprocal_span
        elif shifted_source == self.span:
            return (1 - self.half_reciprocal_span)
        else:
            return (shifted_source * self.reciprocal_span)
    
    def draw(self, source, shape=None):
        """
        Draws a destination point from this jumping distribution given a source
        point.
        
        source: single integer number
        
        returns: either single value (if shape is None) or np.ndarray of given
                 shape.
        """
        return self.minimum + np.random.binomial(self.span,\
            self.p_from_shifted_source(source - self.minimum), size=shape)
    
    @property
    def log_value_constant(self):
        """
        Property storing a constant in the log value of this distribution which
        is independent of the source and destination integers.
        """
        if not hasattr(self, '_log_value_constant'):
            self._log_value_constant = log_gamma(self.span + 1)
        return self._log_value_constant
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF: ln(f(source->destination))
        
        source: either single integer between minimum and maximum or array of
                such values
        destination: if source is a single integer, destination can be a single
                                                    integer or an array of
                                                    integers.
                     if source is an array, destination and source must be
                                            arrays castable to a common shape.
        
        returns: if source is a single number, returns log values in same form
                                               as destination
                 if source and destination are both arrays, returns log values
                                                            in numpy.ndarray of
                                                            same shape as
                                                            destination-source
        """
        shifted_source = source - self.minimum
        shifted_destination = destination - self.minimum
        p_parameter = self.p_from_shifted_source(shifted_source)
        return self.log_value_constant - log_gamma(shifted_destination + 1) -\
            log_gamma(self.span - shifted_destination + 1) +\
            (shifted_destination * np.log(p_parameter)) +\
            ((self.span - shifted_destination) * np.log(1 - p_parameter))
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. It must be implemented by all subclasses.
        """
        return 1
    
    def __eq__(self, other):
        """
        Tests for equality between this jumping distribution and other. All
        subclasses must implement this function.
        
        other: JumpingDistribution with which to check for equality
        
        returns: True or False
        """
        if isinstance(other, BinomialJumpingDistribution):
            return (self.minimum == other.minimum) and\
                (self.maximum == other.maximum)
        else:
            return False
    
    @property
    def is_discrete(self):
        """
        Property storing boolean describing whether this JumpingDistribution
        describes discrete (True) or continuous (False) variable(s).
        """
        return True
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this jumping
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this jumping
               distribution
        """
        group.attrs['class'] = 'BinomialJumpingDistribution'
        group.attrs['minimum'] = self.minimum
        group.attrs['maximum'] = self.maximum
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a BinomialJumpingDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this BinomialJumpingDistribution was saved
        
        returns: a BinomialJumpingDistribution object created from the
                 information in the given group
        """
        try:
            assert group.attrs['class'] == 'BinomialJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "BinomialJumpingDistribution.")
        minimum = group.attrs['minimum']
        maximum = group.attrs['maximum']
        return BinomialJumpingDistribution(minimum, maximum)

