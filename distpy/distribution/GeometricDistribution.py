"""
File: distpy/distribution/GeometricDistribution.py
Author: Keith Tauscher
Date: Oct 15 2019

Description: File containing class representing a geometric distribution.
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, numerical_types
from .Distribution import Distribution

class GeometricDistribution(Distribution):
    """
    Distribution with support on the non-negative integers. It has only one
    parameter, the common ratio between successive probabilities.
    """
    def __init__(self, common_ratio, minimum=0, maximum=None, metadata=None):
        """
        Initializes new GeometricDistribution with given scale.
        
        common_ratio: ratio between successive probabilities
        """
        self.common_ratio = common_ratio
        self.minimum = minimum
        self.maximum = maximum
        self.metadata = metadata
    
    @property
    def common_ratio(self):
        """
        Property storing the common ration between the probability mass
        function of successive integers. Always between 0 and 1 (exclusive)
        """
        if not hasattr(self, '_common_ratio'):
            raise AttributeError("common_ration was referenced before it " +\
                "was set.")
        return self._common_ratio
    
    @common_ratio.setter
    def common_ratio(self, value):
        """
        Setter for the common ratio between the probability mass function of
        successive integers.
        
        value: must be a number between 0 and 1 (exclusive)
        """
        if type(value) in numerical_types:
            if (value > 0.) and (value < 1.):
                self._common_ratio = value
            else:
                raise ValueError("scale given to GeometricDistribution was " +\
                    "not between 0 and 1.")
        else:
            raise ValueError("common_ratio given to GeometricDistribution " +\
                "was not a number.")
    
    @property
    def minimum(self):
        """
        Property storing the lowest integer allowable to be returned by this
        distribution.
        """
        if not hasattr(self, '_minimum'):
            raise AttributeError("minimum referenced before it was set.")
        return self._minimum
    
    @minimum.setter
    def minimum(self, value):
        """
        Setter for the minimum allowable value returned by this distribution.
        
        value: an integer
        """
        if type(value) in int_types:
            self._minimum = value
        else:
            raise TypeError("minimum was set to a non-int.")
    
    @property
    def maximum(self):
        """
        Property storing either the upper limit of this distribution. Can be
        None or an integer greater than minimum.
        """
        if not hasattr(self, '_maximum'):
            raise AttributeError("maximum was referenced before it was set.")
        return self._maximum
    
    @maximum.setter
    def maximum(self, value):
        """
        Setter for the maximum allowable value returned by this distribution.
        
        value: if None, there is no maximum and drawn values can be arbitrarily
                        large
               otherwise, maximum should be an integer greater than minimum
        """
        if type(value) is type(None):
            self._maximum = None
        elif type(value) in int_types:
            if value >= self.minimum:
                self._maximum = value
            else:
                raise ValueError("maximum was not greater than minimum.")
        else:
            raise TypeError("maximum wasn't set to None or an integer.")
    
    @property
    def range(self):
        """
        Property storing the one greater than the distance between minimum and
        maximum.
        """
        if not hasattr(self, '_range'):
            if type(self.maximum) is type(None):
                self._range = None
            else:
                self._range = (self.maximum - self.minimum + 1)
        return self._range
    
    @property
    def constant_in_log_value(self):
        """
        Property storing the portion of the log value which does not depend on
        the point at which the value is computed.
        """
        if not hasattr(self, '_constant_in_log_value'):
            self._constant_in_log_value = np.log(1 - self.common_ratio)
            if type(self.range) is not type(None):
                self._constant_in_log_value -=\
                    np.log(1 - (self.common_ratio ** self.range))
        return self._constant_in_log_value
    
    @property
    def numparams(self):
        """
        Geometric distribution pdf is univariate so numparams always returns 1.
        """
        return 1
    
    @property
    def mean(self):
        """
        Property storing the mean of this distribution.
        """
        if not hasattr(self, '_mean'):
            mean = self.minimum + (self.common_ratio / (1 - self.common_ratio))
            if type(self.maximum) is not type(None):
                mean = mean -\
                    ((self.range * (self.common_ratio ** self.range)) /\
                    (1 - (self.common_ratio ** self.range)))
            self._mean = mean
        return self._mean
    
    @property
    def variance(self):
        """
        Property storing the covariance of this distribution.
        """
        if not hasattr(self, '_variance'):
            roomr = self.common_ratio / (1 - self.common_ratio)
            expected_square = roomr + (2 * (roomr ** 2))
            if type(self.maximum) is not type(None):
                rs = (self.common_ratio ** self.range)
                expected_square = expected_square - ((self.range ** 2) * rs) -\
                    (2 * roomr * self.range * rs) - (roomr * rs) -\
                    (2 * rs * (roomr ** 2))
                expected_square = expected_square /\
                    (1 - (self.common_ratio ** self.range))
            self._variance =\
                expected_square - ((self.mean - self.minimum) ** 2)
        return self._variance
    
    @property
    def log_common_ratio(self):
        """
        Property storing the natural logarithm of the common ratio of
        successive probabilities.
        """
        if not hasattr(self, '_log_common_ratio'):
            self._log_common_ratio = np.log(self.common_ratio)
        return self._log_common_ratio
    
    def draw(self, shape=None, random=rand):
        """
        Draws and returns a value from this distribution using numpy.random.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        random: the random number generator to use (default: numpy.random)
        """
        uniforms = random.uniform(size=shape)
        if type(self.maximum) is type(None):
            log_argument = uniforms
        else:
            log_argument =\
                (1 - (uniforms * (1 - (self.common_ratio ** self.range))))
        return self.minimum +\
            np.floor(np.log(log_argument) / self.log_common_ratio).astype(int)
    
    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        if type(point) in int_types:
            if point >= self.minimum:
                if (type(self.maximum) is not type(None)) and\
                    (point > self.maximum):
                    return -np.inf
                else:
                    return self.constant_in_log_value +\
                        ((point - self.minimum) * self.log_common_ratio)
            else:
                return -np.inf
        else:
            raise TypeError("point given to GeometricDistribution was not " +\
                "an integer.")

    def to_string(self):
        """
        Finds and returns a string version of this GeometricDistribution.
        """
        return "Geometric({:.4g})".format(self.common_ratio)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a GeometricDistribution with the same scale.
        """
        if isinstance(other, GeometricDistribution):
            ratios_close =\
                np.isclose(self.common_ratio, other.common_ratio, atol=1e-6)
            minima_equal = (self.minimum == other.minimum)
            maxima_equal = (self.maximum == other.maximum)
            metadata_equal = self.metadata_equal(other)
            return all([ratios_close, minima_equal, maxima_equal, metadata_equal])
        else:
            return False
    
    @property
    def can_give_confidence_intervals(self):
        """
        In distpy, discrete distributions do not support confidence intervals.
        """
        return False
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return True
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this distribution. The
        only thing to save is the common_ratio.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'GeometricDistribution'
        group.attrs['common_ratio'] = self.common_ratio
        group.attrs['minimum'] = self.minimum
        if type(self.maximum) is not type(None):
            group.attrs['maximum'] = self.maximum
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a GeometricDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a GeometricDistribution object created from the information in
                 the given group
        """
        try:
            assert group.attrs['class'] == 'GeometricDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "GeometricDistribution.")
        metadata = Distribution.load_metadata(group)
        common_ratio = group.attrs['common_ratio']
        minimum = group.attrs['minimum']
        if 'maximum' in group.attrs:
            maximum = group.attrs['maximum']
        else:
            maximum = None
        return GeometricDistribution(common_ratio, minimum=minimum,\
            maximum=maximum, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented. Since this is a discrete distribution, it returns
        False.
        """
        return False 
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. Since this is a discrete distribution, it returns
        False.
        """
        return False
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return GeometricDistribution(self.common_ratio, self.minimum,\
            self.maximum)

