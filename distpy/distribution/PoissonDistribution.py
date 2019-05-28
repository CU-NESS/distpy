"""
File: distpy/distribution/PoissonDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing class representing a Poisson distribution.
"""
import numpy as np
import numpy.random as rand
from scipy.special import gammaln as log_gamma
from ..util import int_types, numerical_types
from .Distribution import Distribution

class PoissonDistribution(Distribution):
    """
    Distribution with support on the nonnegative integers. It has only one
    parameter, the scale, which is both its mean and its variance.
    """
    def __init__(self, scale, metadata=None):
        """
        Initializes new PoissonDistribution with given scale.
        
        scale: mean and variance of distribution (must be positive)
        """
        self.scale = scale
        self.metadata = metadata
    
    @property
    def scale(self):
        """
        Property storing the scale parameter of this Poisson distribution.
        """
        if not hasattr(self, '_scale'):
            raise AttributeError("scale was referenced before it was set.")
        return self._scale
    
    @scale.setter
    def scale(self, value):
        """
        Setter for the scale parameter of this Poisson distribution.
        
        value: positive number, both mean and variance of this distribution
        """
        if type(value) in numerical_types:
            if value > 0:
                self._scale = (value * 1.)
            else:
                raise ValueError("scale given to PoissonDistribution was " +\
                    "not positive.")
        else:
            raise ValueError("scale given to PoissonDistribution was not a " +\
                "number.")
    
    @property
    def numparams(self):
        """
        Poisson distribution pdf is univariate so numparams always returns 1.
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
        return random.poisson(lam=self.scale, size=shape)
    
    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        if type(point) in int_types:
            if point >= 0:
                return (point * np.log(self.scale)) - self.scale -\
                    log_gamma(point + 1)
            else:
                return -np.inf
        else:
            raise TypeError("point given to PoissonDistribution was not an " +\
                "integer.")

    def to_string(self):
        """
        Finds and returns a string version of this PoissonDistribution.
        """
        return "Poisson({:.4g})".format(self.scale)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a PoissonDistribution with the same scale.
        """
        if isinstance(other, PoissonDistribution):
            scale_close =\
                np.isclose(self.scale, other.scale, rtol=1e-6, atol=1e-6)
            metadata_equal = self.metadata_equal(other)
            return (scale_close and metadata_equal)
        else:
            return False
    
    @property
    def can_give_confidence_intervals(self):
        """
        In distpy, confidence intervals are not supported with discrete
        distributions.
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
        return None
    
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
        only thing to save is the scale.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'PoissonDistribution'
        group.attrs['scale'] = self.scale
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a PoissonDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: PoissonDistribution object created from the information in the
                 given group
        """
        try:
            assert group.attrs['class'] == 'PoissonDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "PoissonDistribution.")
        metadata = Distribution.load_metadata(group)
        scale = group.attrs['scale']
        return PoissonDistribution(scale, metadata=metadata)
    
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
        return PoissonDistribution(self.scale)

