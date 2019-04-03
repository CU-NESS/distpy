"""
File: distpy/jumping/GaussianJumpingDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing a jumping distribution which is Gaussian centered
             on the source point with a given covariance.
"""
import numpy as np
import numpy.linalg as npla
import scipy.linalg as scila
from ..util import create_hdf5_dataset, get_hdf5_value, int_types,\
    numerical_types, sequence_types
from .JumpingDistribution import JumpingDistribution

class GaussianJumpingDistribution(JumpingDistribution):
    """
    Class representing a jumping distribution which is centered on the source
    point and has the given covariance.
    """
    def __init__(self, covariance):
        """
        Initializes a GaussianJumpingDistribution with the given covariance
        matrix.
        
        covariance: either single number (if this should be a 1D Gaussian) or
                    square 2D array (if this should be a multivariate Gaussian)
        """
        self.covariance = covariance
    
    @property
    def covariance(self):
        """
        Property storing a 2D numpy.ndarray of covariances.
        """
        if not hasattr(self, '_covariance'):
            raise AttributeError("covariance referenced before it was set.")
        return self._covariance
    
    @covariance.setter
    def covariance(self, value):
        """
        Sets the covariance of this GaussianJumpingDistribution
        
        value: either a single number (if this GaussianJumpingDistribution
               should be 1D) or a square 2D array
        """
        if type(value) in numerical_types:
            self._covariance = np.ones((1, 1)) * value
        elif type(value) in sequence_types:
            value = np.array(value)
            if np.any(np.isnan(value)):
                raise ValueError(("For some reason, there are nan's in the " +\
                    "covariance matrix given to a " +\
                    "GaussianJumpingDistribution, which was:\n{}.").format(\
                    value))
            elif (value.ndim == 2) and (value.shape[0] == value.shape[1]):
                self._covariance = (value + value.T) / 2
            else:
                raise ValueError("covariance didn't have the expected shape.")
        else:
            raise TypeError("covariance was neither a number nor an array.")
        self.inverse_covariance, self.constant_in_log_value # compute stuff
    
    @property
    def inverse_covariance(self):
        """
        Property storing a 2D numpy.ndarray storing the inverse of the matrix
        stored in covariance property.
        """
        if not hasattr(self, '_inverse_covariance'):
            self._inverse_covariance = npla.inv(self.covariance)
        return self._inverse_covariance
    
    @property
    def constant_in_log_value(self):
        """
        Property storing a constant in the log value which is independent of
        both the source and the destination.
        """
        if not hasattr(self, '_constant_in_log_value'):
            self._constant_in_log_value =\
                ((self.numparams * np.log(2 * np.pi)) +\
                npla.slogdet(self.covariance)[1]) / (-2.)
        return self._constant_in_log_value
    
    @property
    def square_root_covariance(self):
        """
        Property storing the square root of the covariance matrix.
        """
        if not hasattr(self, '_square_root_covariance'):
            (eigenvalues, eigenvectors) = npla.eigh(self.covariance)
            if np.any(eigenvalues <= 0):
                raise ValueError(("Something went wrong, causing the square " +\
                    "root of the covariance matrix of this " +\
                    "GaussianJumpingDistribution to have at least one " +\
                    "complex element. The eigenvalues of the covariance " +\
                    "matrix are {!s}.").format(eigenvalues))
            eigenvalues = np.sqrt(eigenvalues)
            self._square_root_covariance =\
                np.dot(eigenvectors * eigenvalues[None,:], eigenvectors.T)
        return self._square_root_covariance
    
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
        random: the random number generator to use (default: numpy.random)
        
        returns: either single value (if distribution is 1D) or array of values
        """
        if self.numparams == 1:
            return random.normal(source, self.standard_deviation, size=shape)
        elif type(shape) is type(None):
            return source + np.dot(self.square_root_covariance,\
                random.normal(0, 1, size=(self.numparams)))
        else:
            if type(shape) in int_types:
                shape = (shape,)
            random_vector = random.normal(0, 1, size=shape+(1, self.numparams))
            return source +\
                np.sum(random_vector * self.square_root_covariance, axis=-1)
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF: ln(f(source->destination))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        difference = (destination - source)
        if self.numparams == 1:
            return (self.constant_in_log_value +\
                (((difference / self.standard_deviation) ** 2) / (-2.)))
        else:
            return (self.constant_in_log_value + (np.dot(difference,\
                np.dot(difference, self.inverse_covariance)) / (-2.)))
    
    def log_value_difference(self, source, destination):
        """
        Computes the log-PDF difference:
        ln(f(source->destination)/f(destination->source))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number difference between one-way log-PDF's
        """
        return 0.
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. It must be implemented by all subclasses.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = self.covariance.shape[0]
        return self._numparams
    
    @property
    def standard_deviation(self):
        """
        Property storing the square root of the variance (in the case that
        numparams == 1). If this Gaussian is multivariate, referencing this
        property will throw a NotImplementedError because the standard
        deviation is not well defined in this case.
        """
        if not hasattr(self, '_standard_deviation'):
            if self.numparams == 1:
                self._standard_deviation = np.sqrt(self.covariance[0,0])
            else:
                raise NotImplementedError("The standard deviation of a " +\
                    "multivariate Gaussian was referenced, but the " +\
                    "standard deviation has no well defined meaning for " +\
                    "multivariate Gaussian distributions.")
        return self._standard_deviation
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: JumpingDistribution with which to check for equality
        
        returns: True or False
        """
        if isinstance(other, GaussianJumpingDistribution):
            if self.numparams == other.numparams:
                return np.allclose(self.covariance, other.covariance,\
                    rtol=1e-12, atol=1e-12)
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
        return False
    
    def fill_hdf5_group(self, group, covariance_link=None):
        """
        Fills the given hdf5 file group with data from this distribution. The
        fact that this is a GaussianJumpingDistribution is saved along with the
        mean and covariance.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'GaussianJumpingDistribution'
        create_hdf5_dataset(group, 'covariance', data=self.covariance,\
            link=covariance_link)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a GaussianJumpingDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this GaussianJumpingDistribution was saved
        
        returns: a GaussianJumpingDistribution object created from the
                 information in the given group
        """
        try:
            assert group.attrs['class'] == 'GaussianJumpingDistribution'
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "GaussianJumpingDistribution.")
        return GaussianJumpingDistribution(get_hdf5_value(group['covariance']))

