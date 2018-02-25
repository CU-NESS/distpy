"""
File: distpy/jumping/AdjacencyJumpingDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing a JumpingDistribution defined
             on the integers (which may or may not have minima and/or maxima)
             which jumps with a specific probability away from the source and
             never jumps more than by 1.
"""
import numpy as np
from ..util import int_types, numerical_types
from .JumpingDistribution import JumpingDistribution

log2 = np.log(2)

class AdjacencyJumpingDistribution(JumpingDistribution):
    """
    Class representing a JumpingDistribution defined on the integers (which may
    or may not have minima and/or maxima) which jumps with a specific
    probability away from the source and never jumps more than by 1.
    """
    def __init__(self, jumping_probability=0.5, minimum=None, maximum=None):
        """
        Initializes an AdjacencyJumpingDistribution with the given jumping
        probability (and extrema, if applicable).
        
        jumping_probability: number between 0 and 1 (exclusive) describing the
                             probability with which the destination is
                             different from the source.
        minimum: if None, no minimum is used
                 if an int, the minimum integer ever drawn by the distribution
        maximum: if None, no maximum is used
                 if an int, the maximum integer ever drawn by the distribution
        """
        self.jumping_probability = jumping_probability
        self.minimum = minimum
        self.maximum = maximum
    
    @property
    def jumping_probability(self):
        """
        Property storing the probability (0<p<1) with which the destination is
        different than the source.
        """
        if not hasattr(self, '_jumping_probability'):
            raise AttributeError("jumping_probability referenced before it " +\
                "was set.")
        return self._jumping_probability
    
    @jumping_probability.setter
    def jumping_probability(self, value):
        """
        Setter for the jumping_probability property.
        
        value: number greater than 0 and less than 1
        """
        if type(value) in numerical_types:
            if (value > 0) and (value < 1):
                self._jumping_probability = value
            else:
                raise ValueError("jumping_probability, jp, doesn't satisfy " +\
                    "0<jp<1.")
        else:
            raise TypeError("jumping_probability was set to a non-number.")
    
    @property
    def log_jumping_probability(self):
        """
        Property storing the natural logarithm of the jumping probability
        """
        if not hasattr(self, '_log_jumping_probability'):
            self._log_jumping_probability = np.log(self.jumping_probability)
        return self._log_jumping_probability
    
    @property
    def log_of_complement_of_jumping_probability(self):
        """
        Property storing the natural logarithm of the complement of the jumping
        probability.
        """
        if not hasattr(self, '_log_of_complement_of_jumping_probability'):
            self._log_of_complement_of_jumping_probability =\
                np.log(1 - self.jumping_probability)
        return self._log_of_complement_of_jumping_probability
    
    @property
    def minimum(self):
        """
        Property storing either None (if this distribution should be able to
        jump towards negative infinity) or the minimum integer this
        distribution should ever draw.
        """
        if not hasattr(self, '_minimum'):
            raise AttributeError("minimum referenced before it was set.")
        return self._minimum
    
    @minimum.setter
    def minimum(self, value):
        """
        Setter for the minimum allowable drawn integer.
        
        value: either None or the minimum allowable integer
        """
        if (value is None) or (type(value) in int_types):
            self._minimum = value
        else:
            raise TypeError("minimum was set to a non-int.")
    
    @property
    def maximum(self):
        """
        Property storing either None (if this distribution should be able to
        jump towards positive infinity) or the maximum integer this
        distribution should ever draw.
        """
        if not hasattr(self, '_maximum'):
            raise AttributeError("maximum referenced before it was set.")
        return self._maximum
    
    @maximum.setter
    def maximum(self, value):
        """
        Setter for the maximum allowable drawn integer.
        
        value: either None or the maximum allowable integer
        """
        if value is None:
            self._maximum = value
        elif type(value) in int_types:
            if (self.minimum  is None) or (value > self.minimum):
                self._maximum = value
            else:
                raise ValueError("maximum wasn't greater than minimum.")
        else:
            raise TypeError("maximum was set to a non-int.")
    
    def draw_single_value(self, source):
        """
        Draws a single value from this distribution.
        
        source: single integer
        
        returns: single integer within 1 of source
        """
        uniform = np.random.rand()
        if uniform < self.jumping_probability:
            if source == self.minimum:
                return (source + 1)
            elif source == self.maximum:
                return (source - 1)
            elif uniform < (self.jumping_probability / 2.):
                return (source - 1)
            else:
                return (source + 1)
        else:
            return source
    
    def draw_shaped_values(self, source, shape):
        """
        Draws arbitrary shape of random values given the source point.
        
        source: a single integer number from which to jump
        shape: tuple of ints describing shape of output
        
        returns: numpy.ndarray of shape shape
        """
        uniform = np.random.rand(*shape)
        jumps = np.where(uniform < self.jumping_probability, 1, 0)
        if source == self.minimum:
            pass
        elif source == self.maximum:
            jumps = -jumps
        else:
            jumps[np.where(uniform < self.jumping_probability / 2.)[0]] = -1
        return source + jumps
    
    def draw(self, source, shape=None):
        """
        Draws a destination point from this jumping distribution given a source
        point.
        
        source: single integer number
        shape: if None, single random value is drawn
               if int n, n random values are drawn
               if tuple of ints, the shape of a numpy.ndarray of destination
                                 values
        
        returns: random values (type/shape determined by shape argument)
        """
        if shape is None:
            return self.draw_single_value(source)
        elif type(shape) in int_types:
            shape = (shape,)
        return self.draw_shaped_values(source, shape)
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF: ln(f(source->destination))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        displacement = destination - source
        if displacement == 0:
            return self.log_of_complement_of_jumping_probability
        elif displacement == 1:
            return_value = self.log_jumping_probability
            if source != self.minimum:
                return_value -= log2
            return return_value
        elif displacement == -1:
            return_value = self.log_jumping_probability
            if source != self.maximum:
                return_value -= log2
            return return_value
        else:
            return -np.inf	
    
    def log_value_difference(self, source, destination):
        """
        Computes the log-PDF difference:
        ln(f(source->destination)/f(destination->source))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number difference between one-way log-PDF's
        """
        displacement = destination - source
        if displacement == 0:
            return 0.
        elif displacement == 1:
            source_is_minimum = (source == self.minimum)
            destination_is_maximum = (destination == self.maximum)
            if source_is_minimum == destination_is_maximum:
                return 0.
            elif source_is_minimum:
                return log2
            else:
                return -log2
        elif displacement == -1:
            source_is_maximum = (source == self.maximum)
            destination_is_minimum = (destination == self.minimum)
            if source_is_maximum == destination_is_minimum:
                return 0.
            elif source_is_maximum:
                return log2
            else:
                return -log2
        else:
            raise ValueError("source and destination could not connected " +\
                "by only a single jump.")
    
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
        if isinstance(other, AdjacencyJumpingDistribution):
            jumping_probabilities_equal = np.isclose(self.jumping_probability,\
                other.jumping_probability, atol=1e-6)
            minima_equal = (self.minimum == other.minimum)
            maxima_equal = (self.maximum == other.maximum)
            return (jumping_probabilities_equal and (minima_equal and\
                maxima_equal))
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
        Fills the given hdf5 file group with information about this
        distribution.
        
        group: hdf5 file group to fill with information about this distribution
        """
        group.attrs['class'] = 'AdjacencyJumpingDistribution'
        group.attrs['jumping_probability'] = self.jumping_probability
        if self.minimum is not None:
            group.attrs['minimum'] = self.minimum
        if self.maximum is not None:
            group.attrs['maximum'] = self.maximum
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an AdjacencyJumpingDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this AdjacencyJumpingDistribution was saved
        
        returns: an AdjacencyJumpingDistribution object created from the
                 information in the given group
        """
        try:
            assert group.attrs['class'] == 'AdjacencyJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain an " +\
                "AdjacencyJumpingDistribution.")
        jumping_probability = group.attrs['jumping_probability']
        if 'minimum' in group.attrs:
            minimum = group.attrs['minimum']
        else:
            minimum = None
        if 'maximum' in group.attrs:
            maximum = group.attrs['maximum']
        else:
            maximum = None
        return\
            AdjacencyJumpingDistribution(jumping_probability, minimum, maximum)

