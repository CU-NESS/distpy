"""
File: distpy/distribution/BernoulliDistribution.py
Author: Keith Tauscher
Date: 15 Oct 2019

Description: File containing class representing a Bernoulli distribution.
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, numerical_types
from .Distribution import Distribution

class BernoulliDistribution(Distribution):
    """
    Distribution with support on the non-negative integers up to a maximum
    numbers. It has only two parameters, probability of sucess and number of
    trials. When probability of success is 1/2, this is the distribution of the
    number of heads flipped in the number of trials given.
    """
    def __init__(self, probability_of_success, metadata=None):
        """
        Initializes new BinomialDistribution with given scale.
        
        probability_of_success: real number in (0, 1)
        metadata: data to store alongside this distribution.
        """
        self.probability_of_success = probability_of_success
        self.metadata = metadata
    
    @property
    def mean(self):
        """
        Property storing the mean of this distribution.
        """
        if not hasattr(self, '_mean'):
            self._mean = self.probability_of_success
        return self._mean
    
    @property
    def variance(self):
        """
        Property storing the variance of this distribution.
        """
        if not hasattr(self, '_variance'):
            self._variance =\
                self.probability_of_success * (1 - self.probability_of_success)
        return self._variance
    
    @property
    def probability_of_success(self):
        """
        Property storing the probability of drawing 1 as opposed to 0.
        """
        if not hasattr(self, '_probability_of_success'):
            raise AttributeError("probability_of_success was referenced " +\
                "before it was set.")
        return self._probability_of_success
    
    @probability_of_success.setter
    def probability_of_success(self, value):
        """
        Setter for the probability of success.
        
        value: real number between 0 and 1 (exclusive)
        """
        if type(value) in numerical_types:
            if (value > 0.) and (value < 1.):
                self._probability_of_success = value
            else:
                raise ValueError("probability_of_success given to " +\
                    "BinomialDistribution was not between 0 and 1.")
        else:
            raise ValueError("probability_of_success given to " +\
                "BinomialDistribution was not a number.")
    
    @property
    def numparams(self):
        """
        Binomial distribution pdf is univariate so numparams always returns 1.
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
        if shape is None:
            none_shape = True
            shape = (1,)
        else:
            none_shape = False
        values = (random.uniform(size=shape) <\
            self.probability_of_success).astype(int)
        if none_shape:
            values = values[0]
        return values
    
    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        if type(point) in int_types:
            if point == 0:
                return np.log(1 - self.probability_of_success)
            elif point == 1:
                return np.log(self.probability_of_success)
            else:
                return -np.inf
        else:
            raise TypeError("point given to BernoulliDistribution was not " +\
                "an integer.")
    
    def to_string(self):
        """
        Finds and returns a string version of this BinomialDistribution.
        """
        return "Bernoulli({:.2g})".format(self.probability_of_success)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a BernoulliDistribution with the same probability_of_success.
        """
        if isinstance(other, BernoulliDistribution):
            p_close = np.isclose(self.probability_of_success,\
                other.probability_of_success, rtol=0, atol=1e-6)
            metadata_equal = self.metadata_equal(other)
            return (p_close and metadata_equal)
        else:
            return False
    
    @property
    def can_give_confidence_intervals(self):
        """
        In distpy, discrete distributions do not support confidence intervals.
        """
        return False
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
        """
        return 0
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        return 1
    
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
        group.attrs['class'] = 'BernoulliDistribution'
        group.attrs['probability_of_success'] = self.probability_of_success
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a BinomialDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a BinomialDistribution object created from the information in
                 the given group
        """
        try:
            assert group.attrs['class'] == 'BernoulliDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "BernoulliDistribution.")
        metadata = Distribution.load_metadata(group)
        probability_of_success = group.attrs['probability_of_success']
        return\
            BernoulliDistribution(probability_of_success, metadata=metadata)
    
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
        return BernoulliDistribution(self.probability_of_success)

