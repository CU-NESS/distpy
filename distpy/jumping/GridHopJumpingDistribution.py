"""
File: distpy/jumping/GridHopJumpingDistribution.py
Author: Keith Tauscher
Date: 22 Sep 2018

Description: File containing class representing a JumpingDistribution defined
             on a grid of integers (which may or may not have minima and/or
             maxima) which jumps with a specific probability away from the
             source and never jumps more than one space.
"""
import numpy as np
from ..util import int_types, numerical_types, sequence_types
from .JumpingDistribution import JumpingDistribution

class GridHopJumpingDistribution(JumpingDistribution):
    """
    Class representing a JumpingDistribution defined on a grid of integers
    (which may or may not have minima and/or maxima) which jumps with a
    specific probability away from the source and never jumps more than one
    space.
    """
    def __init__(self, ndim=2, jumping_probability=0.5, minima=None,\
        maxima=None):
        """
        Initializes an GridHopJumpingDistribution with the given jumping
        probability (and extrema, if applicable).
        
        jumping_probability: number between 0 and 1 (exclusive) describing the
                             probability with which the destination is
                             different from the source.
        minima: sequence of None or integers
        maxima: sequence of None or integers
        """
        self.ndim = ndim
        self.jumping_probability = jumping_probability
        self.minima = minima
        self.maxima = maxima
    
    @property
    def ndim(self):
        """
        Property storing the number of parameters this distribution describes.
        """
        if not hasattr(self, '_ndim'):
            raise AttributeError("ndim was referenced before it was set.")
        return self._ndim
    
    @ndim.setter
    def ndim(self, value):
        """
        Setter for the number of parameters this distribution describes.
        
        value: a positive integer
        """
        if isinstance(value, int):
            if value > 0:
                self._ndim = value
            else:
                raise ValueError("ndim was set to a non-positive integer.")
        else:
            raise TypeError("ndim was set to a non-integer.")
    
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
    def minima(self):
        """
        Property storing either a list of None's or minimal allowable values.
        """
        if not hasattr(self, '_minima'):
            raise AttributeError("minima referenced before it was set.")
        return self._minima
    
    @minima.setter
    def minima(self, value):
        """
        Setter for the minima.
        
        value: sequence of either None or minimal allowable values
        """
        if type(value) in sequence_types:
            if all([((element is None) or isinstance(element, int))\
                for element in value]):
                self._minima = np.array(\
                    [(-np.inf if (element is None) else element)\
                    for element in value])
            else:
                raise ValueError("At least one element of minima was " +\
                    "neither None nor an integer.")
        else:
            raise TypeError("minima was set to a non-sequence.")
    
    @property
    def maxima(self):
        """
        Property storing either a list of None's or maximal allowable values.
        """
        if not hasattr(self, '_maxima'):
            raise AttributeError("maxima referenced before it was set.")
        return self._maxima
    
    @maxima.setter
    def maxima(self, value):
        """
        Setter for the maxima.
        
        value: sequence of either None or maximal allowable values
        """
        if type(value) in sequence_types:
            if all([((element is None) or isinstance(element, int))\
                for element in value]):
                self._maxima = np.array(\
                    [(np.inf if (element is None) else element)\
                    for element in value])
                if np.any(self.maxima <= self.minima):
                    raise ValueError("minima and maxima were not all " +\
                        "compatible with each other.")
            else:
                raise ValueError("At least one element of maxima was " +\
                    "neither None nor an integer.")
        else:
            raise TypeError("maxima was set to a non-sequence.")
    
    @property
    def jumps(self):
        """
        Calculates the nonzero jumps this distribution could ever make.
        
        source: integer tuple of length ndim
        
        returns: array of shape (njumps,ndim) containing the njumps possible
                 jumps from the given source
        """
        if not hasattr(self, '_jumps'):
            self._jumps = np.zeros((2 * self.ndim, self.ndim))
            identity_matrix = np.identity(self.ndim)
            self._jumps[0::2,:] = -identity_matrix
            self._jumps[1::2,:] = identity_matrix
        return self._jumps
    
    def possible_jumps(self, source):
        """
        Finds the indices of jumps which are legal from the given source.
        
        source: integer tuple of length ndim
        
        returns: 1D array of indices into self.jumps which represent legal
                 jumps from the given source
        """
        are_possible = np.ndarray((2 * self.ndim,), dtype=bool)
        are_possible[0::2] = (self.minima < source)
        are_possible[1::2] = (self.maxima > source)
        return np.nonzero(are_possible)[0]
    
    def num_possible_jumps(self, source):
        """
        Calculates the number of possible jumps which can be taken from the
        given source.
        
        source: int tuple of length ndim
        
        return: single positive integer
        """
        return len(self.possible_jumps(source))
    
    def draw_single_value(self, source, random=np.random):
        """
        Draws a single value from this distribution.
        
        source: integer tuple of length ndim
        random: the random number generator to use (default: numpy.random)
        
        returns: integer tuple within 1 of source
        """
        uniform = random.rand()
        if uniform < self.jumping_probability:
            possible_jumps = self.possible_jumps(source)
            num_possible_jumps = len(possible_jumps)
            jump_index = int(np.floor((uniform * num_possible_jumps) /\
                self.jumping_probability))
            return source + self.jumps[possible_jumps[jump_index]]
        else:
            return source
    
    def draw_shaped_values(self, source, shape, random=np.random):
        """
        Draws arbitrary shape of random values given the source point.
        
        source: integer tuple from which to jump
        shape: tuple of ints describing shape of output
        random: the random number generator to use (default: numpy.random)
        
        returns: numpy.ndarray of shape shape
        """
        uniform = random.rand(*shape)
        jump_magnitudes = (uniform < self.jumping_probability).astype(int)
        possible_jumps = self.possible_jumps(source)
        num_possible_jumps = len(possible_jumps)
        jump_indices = np.floor((uniform * num_possible_jumps) /\
            self.jumping_probability).astype(int) % num_possible_jumps
        jumps = jump_magnitudes[...,np.newaxis] *\
            self.jumps[possible_jumps[jump_indices],:]
        source_slice = ((np.newaxis,) * len(shape)) + (slice(None),)
        return source[source_slice] + jumps
    
    def draw(self, source, shape=None, random=np.random):
        """
        Draws a destination point from this jumping distribution given a source
        point.
        
        source: single integer number
        shape: if None, single random value is drawn
               if int n, n random values are drawn
               if tuple of ints, the shape of a numpy.ndarray of destination
                                 values
        random: the random number generator to use (default: numpy.random)
        
        returns: random values (type/shape determined by shape argument)
        """
        if shape is None:
            return self.draw_single_value(source, random=random)
        if type(shape) in int_types:
            shape = (shape,)
        return self.draw_shaped_values(source, shape, random=random)
    
    def is_allowable(self, point):
        """
        Finds whether the given point is between the minima and maxima.
        
        point: int tuple of length ndim
        
        returns: True or False
        """
        return (np.all(self.minima <= point) and np.all(self.maxima >= point))
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF: ln(f(source->destination))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        if (not self.is_allowable(source)) or\
            (not self.is_allowable(destination)):
            return -np.inf
        displacement = destination - source
        taxi_cab_distance = np.sum(np.abs(displacement))
        if taxi_cab_distance == 0:
            return self.log_of_complement_of_jumping_probability
        elif taxi_cab_distance == 1:
            return self.log_jumping_probability -\
                np.log(self.num_possible_jumps(source))
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
        if (not self.is_allowable(source)) or\
            (not self.is_allowable(destination)):
            raise ValueError("Either source or destination is not between " +\
                "minima and maxima.")
        displacement = destination - source
        taxi_cab_distance = np.sum(np.abs(displacement))
        if taxi_cab_distance == 0:
            return 0
        elif taxi_cab_distance == 1:
            return np.log(self.num_possible_jumps(destination)) -\
                np.log(self.num_possible_jumps(source))
        else:
            raise ValueError("source and destination could not be " +\
                "connected by a single jump.")
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. It must be implemented by all subclasses.
        """
        return self.ndim
    
    def __eq__(self, other):
        """
        Tests for equality between this jumping distribution and other. All
        subclasses must implement this function.
        
        other: JumpingDistribution with which to check for equality
        
        returns: True or False
        """
        if isinstance(other, GridHopJumpingDistribution):
            if self.ndim == other.ndim:
                if np.all(self.minima == other.minima) and\
                    np.all(self.maxima == other.maxima):
                    return np.isclose(self.jumping_probability,\
                        other.jumping_probability, atol=1e-6)
                else:
                    return False
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
        group.attrs['class'] = 'GridHopJumpingDistribution'
        group.attrs['jumping_probability'] = self.jumping_probability
        group.attrs['ndim'] = self.ndim
        group.create_dataset('minima', data=self.minima)
        group.create_dataset('maxima', data=self.maxima)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an GridHopJumpingDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this GridHopJumpingDistribution was saved
        
        returns: an GridHopJumpingDistribution object created from the
                 information in the given group
        """
        try:
            assert group.attrs['class'] == 'GridHopJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain an " +\
                "GridHopJumpingDistribution.")
        ndim = group.attrs['ndim']
        jumping_probability = group.attrs['jumping_probability']
        minima = [None if (minimum == -np.inf) else minimum\
            for minimum in group['minima'].value]
        maxima = [None if (maximum == np.inf) else maximum\
            for maximum in group['maxima'].value]
        return GridHopJumpingDistribution(ndim=ndim,\
            jumping_probability=jumping_probability, minima=minima,\
            maxima=maxima)

