"""
File: distpy/distribution/DiscreteUniformDistribution.py
Author: Keith Tauscher
Date: 16 Jun 2018

Description: File containing a class representing a discrete uniform
             distribution.
"""
import numpy as np
import numpy.random as rand
from ..util import int_types
from .Distribution import Distribution

class DiscreteUniformDistribution(Distribution):
    """
    Class representing a discrete uniform distribution. Uniform distributions
    are the least informative possible distributions (on a given support) and
    are thus ideal when ignorance abound.
    """
    def __init__(self, low, high=0, metadata=None):
        """
        Creates a new DiscreteUniformDistribution with the given range.
        
        low: lower limit of pdf (defaults to 0)
        high: upper limit of pdf (if not given, left endpoint is zero and right
              endpoint is low)
        """
        if (type(low) in int_types) and (type(high) in int_types):
            if low <= high:
                self.low = low
                self.high = high
            else:
                self.low = high
                self.high = low
        else:
            raise ValueError('Either the low or high endpoint of a ' +\
                'UniformDistribution was not of a numerical type.')
        self._log_P = - np.log(self.high - self.low + 1)
        self.metadata = metadata
    
    @property
    def numparams(self):
        """
        Only univariate uniform distributions are included here so numparams
        always returns 1.
        """
        return 1
    
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
        return random.randint(self.low, high=self.high+1, size=shape)
    
    def log_value(self, point):
        """
       Evaluates and returns the log of the value of this distribution when
        the variable is value.
        
        point: numerical value of the variable
        """
        if (abs(point - int(round(point))) < 1e-9) and (point >= self.low) and\
            (point <= self.high):
            return self._log_P
        return -np.inf
    
    def to_string(self):
        """
        Finds and returns a string representation of this distribution.
        """
        return "DiscreteUniform({0:.2g}, {1:.2g})".format(self.low, self.high)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a DiscreteUniformDistribution with the same high and low and
        False otherwise.
        """
        if isinstance(other, DiscreteUniformDistribution):
            low_equal = (self.low == other.low)
            high_equal = (self.high == other.high)
            metadata_equal = self.metadata_equal(other)
            return all([low_equal, high_equal, metadata_equal])
        else:
            return False
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
        """
        return self.low
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        return self.high
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return True
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data from this distribution. All
        that needs to be saved is the class name and high and low values.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'DiscreteUniformDistribution'
        group.attrs['low'] = self.low
        group.attrs['high'] = self.high
        if save_metadata:
            self.save_metadata(group)
   
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a UniformDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: UniformDistribution object created from the information in the
                 given group
        """
        try:
            assert group.attrs['class'] == 'DiscreteUniformDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "DiscreteUniformDistribution.")
        metadata = Distribution.load_metadata(group)
        low = group.attrs['low']
        high = group.attrs['high']
        return\
            DiscreteUniformDistribution(low=low, high=high, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented. Since this distribution is discrete, it returns
        False.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. Since this distribution is discrete, it returns
        False.
        """
        return False
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return DiscreteUniformDistribution(self.low, self.high)

