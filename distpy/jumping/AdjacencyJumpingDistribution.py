"""
Module containing class representing a 1D distribution that jumps from integers
to integers. Its PMF is represented by: $$f(x,y) = \\begin{cases}\
1-p & y=x \\\\ p/2 & y=x-1\\wedge x\\ne \\text{max} \\\\\
p & y=x-1\\wedge x=\\text{max}\\\\ p/2 & y=x+1\\wedge x\\ne \\text{min}\\\\\
p & y=x+1\\wedge x=\\text{min}\\\\ 0 & \\text{otherwise}\\end{cases}$$

**File**: $DISTPY/distpy/jumping/AdjacencyJumpingDistribution.py  
**Author**: Keith Tauscher  
**Date**: 3 Jul 2021
"""
import numpy as np
from ..util import int_types, numerical_types
from .JumpingDistribution import JumpingDistribution

log2 = np.log(2)

class AdjacencyJumpingDistribution(JumpingDistribution):
    """
    Class representing a 1D distribution that jumps from integers to integers.
    Its PMF is represented by: $$f(x,y) = \\begin{cases} 1-p & y=x \\\\\
    p/2 & y=x-1\\wedge x\\ne \\text{max} \\\\ p & y=x-1\\wedge x=\\text{max}\
    \\\\ p/2 & y=x+1\\wedge x\\ne \\text{min}\\\\ p &\
    y=x+1\\wedge x=\\text{min}\\\\ 0 & \\text{otherwise}\\end{cases}$$
    """
    def __init__(self, jumping_probability=0.5, minimum=None, maximum=None):
        """
        Initializes an `AdjacencyJumpingDistribution` with the given jumping
        probability (and extrema, if applicable).
        
        Parameters
        ----------
        jumping_probability : float
            number between 0 and 1 (exclusive) describing the probability with
            which the destination is different from the source.
        minimum : int or None
            - if None, no minimum is used
            - if an int, the minimum integer ever drawn by the distribution
        maximum : int or None
            - if None, no maximum is used
            - if an int, the maximum integer ever drawn by the distribution
        """
        self.jumping_probability = jumping_probability
        self.minimum = minimum
        self.maximum = maximum
    
    @property
    def jumping_probability(self):
        """
        The probability, \\(0<p<1\\) with which the destination is different
        than the source.
        """
        if not hasattr(self, '_jumping_probability'):
            raise AttributeError("jumping_probability referenced before it " +\
                "was set.")
        return self._jumping_probability
    
    @jumping_probability.setter
    def jumping_probability(self, value):
        """
        Setter for `AdjacencyJumpingDistribution.jumping_probability`.
        
        Parameters
        ----------
        value : float
            number greater than 0 and less than 1
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
        The natural logarithm of the jumping probability, \\(\\ln{p}\\).
        """
        if not hasattr(self, '_log_jumping_probability'):
            self._log_jumping_probability = np.log(self.jumping_probability)
        return self._log_jumping_probability
    
    @property
    def log_of_complement_of_jumping_probability(self):
        """
        The natural logarithm of the complement of the jumping probability,
        \\(\\ln{(1-p)}\\).
        """
        if not hasattr(self, '_log_of_complement_of_jumping_probability'):
            self._log_of_complement_of_jumping_probability =\
                np.log(1 - self.jumping_probability)
        return self._log_of_complement_of_jumping_probability
    
    @property
    def minimum(self):
        """
        Either None (if this distribution should be able to jump towards
        negative infinity) or the minimum integer this distribution should ever
        draw.
        """
        if not hasattr(self, '_minimum'):
            raise AttributeError("minimum referenced before it was set.")
        return self._minimum
    
    @minimum.setter
    def minimum(self, value):
        """
        Setter for `AdjacencyJumpingDistribution.minimum`.
        
        Parameters
        ----------
        value : int or None
            - if None, no minimum is used
            - if an int, the minimum integer ever drawn by the distribution
        """
        if (type(value) is type(None)) or (type(value) in int_types):
            self._minimum = value
        else:
            raise TypeError("minimum was set to a non-int.")
    
    @property
    def maximum(self):
        """
        Either None (if this distribution should be able to jump towards
        infinity) or the maximum integer this distribution should ever draw.
        """
        if not hasattr(self, '_maximum'):
            raise AttributeError("maximum referenced before it was set.")
        return self._maximum
    
    @maximum.setter
    def maximum(self, value):
        """
        Setter for `AdjacencyJumpingDistribution.maximum`.
        
        Parameters
        ----------
        value : int or None
            - if None, no maximum is used
            - if an int, the maximum integer ever drawn by the distribution
        """
        if type(value) is type(None):
            self._maximum = value
        elif type(value) in int_types:
            if (type(self.minimum) is type(None)) or (value > self.minimum):
                self._maximum = value
            else:
                raise ValueError("maximum wasn't greater than minimum.")
        else:
            raise TypeError("maximum was set to a non-int.")
    
    def draw_single_value(self, source, random=np.random):
        """
        Draws a single value from this distribution.
        
        Parameters
        ----------
        source : int
            single starting point
        random : numpy.random.RandomState
            the random number generator to use (default: `numpy.random`)
        
        Returns
        -------
        destination : int
            single int satisfying \\((y-x)\\in\\{-1,0,1\\}\\)
        """
        uniform = random.rand()
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
    
    def draw_shaped_values(self, source, shape, random=np.random):
        """
        Draws arbitrary shape of random values given the source point.
        
        Parameters
        ----------
        source : int
            a single integer number from which to jump
        shape : tuple
            tuple of ints \\((n_1,n_2,\\ldots,n_k)\\),
            \\(\\prod_{m=1}^kn_m\\) destinations are returned as a
            `numpy.ndarray` of shape \\((n_1,n_2,\\ldots,n_k)\\)
        random : numpy.random.RandomState
            the random number generator to use (default: `numpy.random`)
        
        Returns
        -------
        destinations : int or numpy.ndarray
            drawn destination(s)
        """
        uniform = random.rand(*shape)
        jumps = np.where(uniform < self.jumping_probability, 1, 0)
        if source == self.minimum:
            pass
        elif source == self.maximum:
            jumps = -jumps
        else:
            jumps[np.where(uniform < self.jumping_probability / 2.)[0]] = -1
        return source + jumps
    
    def draw(self, source, shape=None, random=np.random):
        """
        Draws random value(s) given the source point.
        
        Parameters
        ----------
        source : int
            a single integer number from which to jump
        shape : None or int or tuple
            - if None, a single int destination is returned
            - if int \\(n\\), \\(n\\) destinations are returned in a 1D
            `numpy.ndarray` of length \\(n\\)
            - if tuple of ints \\((n_1,n_2,\\ldots,n_k)\\),
            \\(\\prod_{m=1}^kn_m\\) destinations are returned as a
            `numpy.ndarray` of shape \\((n_1,n_2,\\ldots,n_k)\\)
        random : numpy.random.RandomState
            the random number generator to use (default: `numpy.random`)
        
        Returns
        -------
        destinations : int or numpy.ndarray
            drawn destination(s)
        """
        if type(shape) is type(None):
            return self.draw_single_value(source, random=random)
        if type(shape) in int_types:
            shape = (shape,)
        return self.draw_shaped_values(source, shape, random=random)
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF of jumping from `source` to `destination`. It must
        be implemented by all subclasses.
        
        Parameters
        ----------
        source : int
            source point
        destination : int
            destination point
        
        Returns
        -------
        log_pdf : float
            if the distribution is \\(f(x,y)=\\text{Pr}[y|x]\\), `source` is
            \\(x\\) and `destination` is \\(y\\), then `log_pdf` is given by
            \\(\\ln{f(x,y)}\\)
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
        Computes the difference in the log-PDF of jumping from `source` to
        `destination` and of jumping from `destination` to `source`.
        
        Parameters
        ----------
        source : int
            source point
        destination : int
            destination point
        
        Returns
        -------
        log_pdf_difference : float
            if the distribution is \\(f(x,y)=\\text{Pr}[y|x]\\), `source` is
            \\(x\\) and `destination` is \\(y\\), then `log_pdf_difference` is
            given by \\(\\ln{f(x,y)}-\\ln{f(y,x)}\\)
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
        The integer number of parameters described by this distribution. Since
        this distribution is univariate, this property is 1.
        """
        return 1
    
    def __eq__(self, other):
        """
        Tests for equality between this jumping distribution and other.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is an `AdjacencyJumpingDistribution`
            with the same `AdjacencyJumpingDistribution.minimum`,
            `AdjacencyJumpingDistribution.maximum`, and
            `AdjacencyJumpingDistribution.jumping_probability`
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
        Boolean describing whether this `AdjacencyJumpingDistribution`. Since
        it exists, on a grid, this is always True.
        """
        return True
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this
        distribution.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill with information about this distribution
        """
        group.attrs['class'] = 'AdjacencyJumpingDistribution'
        group.attrs['jumping_probability'] = self.jumping_probability
        if type(self.minimum) is not type(None):
            group.attrs['minimum'] = self.minimum
        if type(self.maximum) is not type(None):
            group.attrs['maximum'] = self.maximum
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an `AdjacencyJumpingDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which
            `AdjacencyJumpingDistribution.fill_hdf5_group` was called on
        
        Returns
        -------
        loaded : `AdjacencyJumpingDistribution`
            an `AdjacencyJumpingDistribution` object created from the
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

