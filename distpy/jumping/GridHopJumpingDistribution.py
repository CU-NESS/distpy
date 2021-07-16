"""
Module containing class representing a jumping distribution that exists on a
discrete grid. It can only yield jumps that go to neighbor points of the source
or don't move at all.

**File**: $DISTPY/distpy/jumping/GridHopJumpingDistribution.py  
**Author**: Keith Tauscher  
**Date**: 11 Jul 2021
"""
import numpy as np
from ..util import int_types, numerical_types, sequence_types,\
    create_hdf5_dataset, get_hdf5_value
from .JumpingDistribution import JumpingDistribution

class GridHopJumpingDistribution(JumpingDistribution):
    """
    Class representing a jumping distribution that exists on a discrete grid.
    It can only yield jumps that go to neighbor points of the source or don't
    move at all.
    """
    def __init__(self, ndim=2, jumping_probability=0.5, minima=None,\
        maxima=None):
        """
        Initializes a `GridHopJumpingDistribution` with the given jumping
        probability (and extrema, if applicable).
        
        Parameters
        ----------
        jumping_probability : float
            number between 0 and 1 (exclusive) describing the probability with
            which the destination is different from the source.
        minima : sequence
            sequence of None or integers describing grid mins
        maxima : sequence
            sequence of None or integers describing grid maxes
        """
        self.ndim = ndim
        self.jumping_probability = jumping_probability
        self.minima = minima
        self.maxima = maxima
    
    @property
    def ndim(self):
        """
        The number of parameters this distribution describes.
        """
        if not hasattr(self, '_ndim'):
            raise AttributeError("ndim was referenced before it was set.")
        return self._ndim
    
    @ndim.setter
    def ndim(self, value):
        """
        Setter for `GridHopJumpingDistribution.ndim`.
        
        Parameters
        ----------
        value : int
            an integer greater than 1
        """
        if type(value) in int_types:
            if value > 0:
                if value > 1:
                    self._ndim = value
                else:
                    raise ValueError("The GridHopJumpingDistribution class " +\
                        "should not be initialized with only one parameter " +\
                        "as the AdjacencyJumpingDistribution performs the " +\
                        "exact same task more efficiently.")
            else:
                raise ValueError("ndim was set to a non-positive integer.")
        else:
            print('type(value)={}'.format(type(value)))
            raise TypeError("ndim was set to a non-integer.")
    
    @property
    def jumping_probability(self):
        """
        The probability, \\(0<p<1\\), with which the destination is different
        than the source.
        """
        if not hasattr(self, '_jumping_probability'):
            raise AttributeError("jumping_probability referenced before it " +\
                "was set.")
        return self._jumping_probability
    
    @jumping_probability.setter
    def jumping_probability(self, value):
        """
        Setter for `GridHopJumpingDistribution.jumping_probability`.
        
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
        The natural logarithm of the jumping probability, \\(\\ln{p}\\)
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
    def minima(self):
        """
        A list of either None's or minimal allowable values.
        """
        if not hasattr(self, '_minima'):
            raise AttributeError("minima referenced before it was set.")
        return self._minima
    
    @minima.setter
    def minima(self, value):
        """
        Setter for `GridHopJumpingDistribution.minima`.
        
        Parameters
        ----------
        value : sequence
            sequence of either None or minimal allowable values for each
            parameter
        """
        if type(value) in sequence_types:
            if all([((type(element) is type(None)) or\
                (type(element) in int_types)) for element in value]):
                self._minima = np.array(\
                    [(-np.inf if (type(element) is type(None)) else element)\
                    for element in value])
            else:
                raise ValueError("At least one element of minima was " +\
                    "neither None nor an integer.")
        else:
            raise TypeError("minima was set to a non-sequence.")
    
    @property
    def maxima(self):
        """
        A list of either None's or maximal allowable values.
        """
        if not hasattr(self, '_maxima'):
            raise AttributeError("maxima referenced before it was set.")
        return self._maxima
    
    @maxima.setter
    def maxima(self, value):
        """
        Setter for `GridHopJumpingDistribution.maxima`.
        
        Parameters
        ----------
        value : sequence
            sequence of either None or minimal allowable values for each
            parameter
        """
        if type(value) in sequence_types:
            if all([((type(element) is type(None)) or\
                (type(element) in int_types)) for element in value]):
                self._maxima = np.array(\
                    [(np.inf if (type(element) is type(None)) else element)\
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
        The nonzero jumps this distribution could ever make as a
        \\(2N\\times N\\) matrix.
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
        
        Parameters
        ----------
        source : sequence
            integer sequence of length `GridHopJumpingDistribution.ndim`
        
        Returns
        -------
        returns : numpy.ndarray
            1D array of indices into `GridHopJumpingDistribution.jumps` which
            represent legal jumps from the given source
        """
        are_possible = np.ndarray((2 * self.ndim,), dtype=bool)
        are_possible[0::2] = (self.minima < source)
        are_possible[1::2] = (self.maxima > source)
        return np.nonzero(are_possible)[0]
    
    def num_possible_jumps(self, source):
        """
        Finds the number of possible jumps which can be taken from the given
        source.
        
        Parameters
        ----------
        source : sequence
            int sequence of length `GridHopJumpingDistribution.ndim`
        
        Returns
        -------
        num_jumps : int
            single positive integer number of legal nonzero jumps
        """
        return len(self.possible_jumps(source))
    
    def draw_single_value(self, source, random=np.random):
        """
        Draws a single value from this distribution.
        
        Parameters
        ----------
        source : sequence
            integer sequence of length `GridHopJumpingDistribution.ndim`
        random : numpy.random.RandomState
            the random number generator to use (default: numpy.random)
        
        Returns
        -------
        destination : sequence
            single integer tuple within 1 of source
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
        
        Parameters
        ----------
        source : sequence
            integer sequence of length `GridHopJumpingDistribution.ndim`
        shape : tuple
            shape of `destinations[...,index]`
        random : numpy.random.RandomState
            the random number generator to use (default: numpy.random)
        
        Returns
        -------
        destination : numpy.ndarray
            array of shape `shape+(GridHopJumpingDistribution.ndim,)`
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
        point. Must be implemented by any base class.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this `JumpingDistribution` is univariate, source should be
            a single number
            - otherwise, source should be `numpy.ndarray` of shape (numparams,)
        shape : None or int or tuple
            - if None, a single destination is returned as a 1D `numpy.ndarray`
            describing the coordinates of the destination is returned as a 2D
            `numpy.ndarray` is returned whose shape is \\((n,p)\\)
            - if tuple of ints \\((n_1,n_2,\\ldots,n_k)\\),
            \\(\\prod_{m=1}^kn_m\\) destinations are returned as a
            `numpy.ndarray` of shape \\((n_1,n_2,\\ldots,n_k,p)\\) is returned
        random : numpy.random.RandomState
            the random number generator to use (default: `numpy.random`)
        
        Returns
        -------
        drawn : numpy.ndarray
            destination point(s) drawn
        """
        if type(shape) is type(None):
            return self.draw_single_value(source, random=random)
        if type(shape) in int_types:
            shape = (shape,)
        return self.draw_shaped_values(source, shape, random=random)
    
    def is_allowable(self, point):
        """
        Finds whether the given point is between the
        `GridHopJumpingDistribution.minima` and
        `GridHopJumpingDistribution.maxima`.
        
        Parameters
        ----------
        point : sequence
            int tuple of length `GridHopJumpingDistribution.ndim`
        
        Returns
        -------
        result : bool
            True if and only if every value in `point` is between its
            corresponding minimum and maximum
        """
        return (np.all(self.minima <= point) and np.all(self.maxima >= point))
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF of jumping from `source` to `destination`.
        
        Parameters
        ----------
        source : numpy.ndarray
            if this distribution describes \\(p\\) parameters, `source` must
            be a 1D `numpy.ndarray` of length \\(p\\)
        destination : numpy.ndarray
            if this distribution describes \\(p\\) parameters, `destination`
            must be a 1D `numpy.ndarray` of length \\(p\\)
        
        Returns
        -------
        log_pdf : float
            if the distribution is \\(f(\\boldsymbol{x},\\boldsymbol{y})=\
            \\text{Pr}[\\boldsymbol{y}|\\boldsymbol{x}]\\), `source` is
            \\(\\boldsymbol{x}\\) and `destination` is \\(\\boldsymbol{y}\\),
            then `log_pdf` is given by
            \\(\\ln{f(\\boldsymbol{x},\\boldsymbol{y})}\\)
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
        Computes the difference in the log-PDF of jumping from `source` to
        `destination` and of jumping from `destination` to `source`.
        
        Parameters
        ----------
        source : numpy.ndarray
            if this distribution describes \\(p\\) parameters, `source` must
            be a 1D `numpy.ndarray` of length \\(p\\)
        destination : numpy.ndarray
            if this distribution describes \\(p\\) parameters, `destination`
            must be a 1D `numpy.ndarray` of length \\(p\\)
        
        Returns
        -------
        log_pdf_difference : float
            if the distribution is \\(f(\\boldsymbol{x},\\boldsymbol{y})=\
            \\text{Pr}[\\boldsymbol{y}|\\boldsymbol{x}]\\), `source` is
            \\(\\boldsymbol{x}\\) and `destination` is \\(\\boldsymbol{y}\\),
            then `log_pdf_difference` is given by \\(\\ln{f(\\boldsymbol{x},\
            \\boldsymbol{y})}-\\ln{f(\\boldsymbol{y},\\boldsymbol{x})}\\)
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
        The integer number of parameters described by this distribution. Same
        as `GridHopJumpingDistribution.ndim`
        """
        return self.ndim
    
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
            True if and only if `other` is an `GridHopJumpingDistribution`
            with the same `GridHopJumpingDistribution.minima` and
            `GridHopJumpingDistribution.maxima`, and
            `GridHopJumpingDistribution.jumping_probability`
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
        Boolean describing whether this `GridHopJumpingDistribution` describes
        discrete (True) or continuous (False) variable(s). Since this is a
        discrete distribution, it is always True.
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
        group.attrs['class'] = 'GridHopJumpingDistribution'
        group.attrs['jumping_probability'] = self.jumping_probability
        group.attrs['ndim'] = self.ndim
        create_hdf5_dataset(group, 'minima', data=self.minima)
        create_hdf5_dataset(group, 'maxima', data=self.maxima)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `GridHopJumpingDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which
            `GridHopJumpingDistribution.fill_hdf5_group` was called on
        
        Returns
        -------
        loaded : `GridHopJumpingDistribution`
            a `GridHopJumpingDistribution` object loaded from the given group
        """
        try:
            assert group.attrs['class'] == 'GridHopJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain an " +\
                "GridHopJumpingDistribution.")
        ndim = group.attrs['ndim']
        jumping_probability = group.attrs['jumping_probability']
        minima = [None if (minimum == -np.inf) else minimum\
            for minimum in get_hdf5_value(group['minima'])]
        maxima = [None if (maximum == np.inf) else maximum\
            for maximum in get_hdf5_value(group['maxima'])]
        return GridHopJumpingDistribution(ndim=ndim,\
            jumping_probability=jumping_probability, minima=minima,\
            maxima=maxima)

