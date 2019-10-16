"""
File: distpy/distribution/GaussianDistribution.py
Author: Keith Tauscher
Date: Oct 15 2019

Description: File containing class representing Gaussian distribution
             (univariate or multivariate).
"""
from __future__ import division
import numpy as np
import numpy.random as rand
import numpy.linalg as npla
import scipy.linalg as scila
from scipy.special import erfinv
from ..util import numerical_types, int_types, sequence_types,\
    create_hdf5_dataset, get_hdf5_value
from .Distribution import Distribution

natural_log_two_pi = np.log(2 * np.pi)

class GaussianDistribution(Distribution):
    """
    A multivariate (or univariate) Gaussian distribution. The classic. Useful
    when some knowledge of the parameters exists and those parameters can be
    any real number.
    """
    def __init__(self, mean, covariance, metadata=None):
        """
        Initializes either a univariate or a multivariate GaussianDistribution.

        mean the mean must be either a number (if univariate)
                                  or a 1D array (if multivariate)
        covariance the covariance must be either a number (if univariate)
                   or a 2D array (if multivariate) final covariance used is
                   average of this 2D array and its transpose
        """
        self.internal_mean = mean
        self.covariance = covariance
        self.metadata = metadata
    
    @staticmethod
    def combine(*distributions):
        """
        Combines many GaussianDistribution objects into one by concatenating
        their means and covariance matrices.
        
        *distributions: a sequence of GaussianDistribution objects to combine
        
        returns: a single GaussianDistribution object 
        """
        if all([isinstance(distribution, GaussianDistribution)\
            for distribution in distributions]):
            new_mean = np.concatenate([distribution.internal_mean.A[0]\
                for distribution in distributions])
            new_covariance = scila.block_diag(*[distribution.covariance.A\
                for distribution in distributions])
            return GaussianDistribution(new_mean, new_covariance)
        else:
            raise TypeError("At least one of the distributions given to the " +\
                "GaussianDistribution class' combine function was not a " +\
                "GaussianDistribution.")
    
    @property
    def internal_mean(self):
        """
        Property storing the mean of this GaussianDistribution in matrix form.
        """
        if not hasattr(self, '_internal_mean'):
            raise AttributeError("internal_mean was referenced before it " +\
                "was set.")
        return self._internal_mean
    
    @internal_mean.setter
    def internal_mean(self, value):
        """
        Setter for the mean of this distribution
        
        value: either a single number (if univariate) or a 1D numpy.ndarray of
               length numparams
        """
        if type(value) in numerical_types:
            value = [value]
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim != 1:
                raise ValueError("The mean of a GaussianDistribution was " +\
                    "not 1 dimensional.")
            elif value.size == 0:
                raise ValueError("The mean of a GaussianDistribution was " +\
                    "set to something like an empty array.")
            else:
                self._internal_mean = np.matrix(value)
        else:
            raise ValueError("The mean of a GaussianDistribution is not of " +\
                "a recognizable type.")
    
    @property
    def covariance(self):
        """
        Property storing the covariance matrix of this Gaussian.
        """
        if not hasattr(self, '_covariance'):
            raise AttributeError("covariance was referenced before it was " +\
                "set.")
        return self._covariance
    
    @covariance.setter
    def covariance(self, value):
        """
        Setter for the covariance matrix of this Gaussian.
        
        value: if mean has length 1, then this can be a single number
                                     representing the variance
               otherwise, this should be a square positive definite matrix of
                          rank numparams or a 1D array of variances (in which
                          case the variates are assumed independent)
        """
        if type(value) in numerical_types:
            if self.numparams == 1:
                self._covariance = np.matrix([[value]])
            else:
                raise TypeError("covariance was set to a number even " +\
                    "though this Gaussian is multi-dimensional.")
        elif type(value) in sequence_types:
            value = np.array(value)
            if np.any(np.isnan(value)):
                raise ValueError(("For some reason, there are nan's in the " +\
                    "covariance matrix given to a GaussianDistribution, " +\
                    "which was:\n{}.").format(value))
            elif value.shape == (self.numparams,):
                self._covariance = np.matrix(np.diag(value))
            elif value.shape == ((self.numparams,) * 2):
                self._covariance = np.matrix((value + value.T) / 2)
            else:
                raise ValueError("The covariance given to a " +\
                    "GaussianDistribution was not castable to an array of " +\
                    "the correct shape. It should be a square shape with " +\
                    "the same side length as length of mean.")
        else:
            raise ValueError("The mean of a GaussianDistribution is " +\
                "array-like but its covariance isn't matrix like.")
        self.square_root_covariance
    
    @property
    def mean(self):
        """
        Property storing the mean of this distribution.
        """
        if not hasattr(self, '_mean'):
            if self.numparams == 1:
                self._mean = self.internal_mean.A[0,0]
            else:
                self._mean = self.internal_mean.A[0]
        return self._mean
    
    @property
    def variance(self):
        """
        Property storing the covariance of this distribution.
        """
        if not hasattr(self, '_variance'):
            if self.numparams == 1:
                self._variance = self.covariance.A[0,0]
            else:
                self._variance = self.covariance.A
        return self._variance
    
    @property
    def log_determinant_covariance(self):
        """
        Property storing the natural logarithm of the determinant of the
        covariance matrix.
        """
        if not hasattr(self, '_log_determinant_covariance'):
            self._log_determinant_covariance = npla.slogdet(self.covariance)[1]
        return self._log_determinant_covariance
    
    @property
    def inverse_covariance(self):
        """
        Property storing the inverse of the covariance.
        """
        if not hasattr(self, '_inverse_covariance'):
            self._inverse_covariance = npla.inv(self.covariance)
        return self._inverse_covariance

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters which this
        GaussianDistribution describes (same as dimension of mean and
        covariance).
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.internal_mean.A[0])
        return self._numparams
    
    def __add__(self, other):
        """
        Adds the Gaussian random variate described by this distribution
        to the given object.

        other: if other is a constant, the returned Gaussian is the same as
                                       this one with other added to the mean
                                       and the same covariance
               if other is a 1D numpy.ndarray, it must be of the same length
                                               as the dimension of this
                                               GaussianDistribution. In this
                                               case, the returned
                                               GaussianDistribution is the
                                               distribution of the sum of this
                                               Gaussian variate with other
               if other is a GaussianDistribution, it must have the same number
                                                   of parameters as this one
        
        returns: GaussianDistribution representing the addition of this
                 Gaussian variate with other
        """
        if isinstance(other, GaussianDistribution):
            new_mean = self.internal_mean.A[0] + other.internal_mean.A[0]
            new_covariance = self.covariance.A + other.covariance.A
        elif type(other) in [list, tuple, np.ndarray]:
            other = np.array(other)
            if other.ndim == 1:
                if len(other) == self.numparams:
                    new_mean = self.internal_mean.A[0] + other
                    new_covariance = self.covariance.A
                else:
                    raise ValueError("Cannot multiply Gaussian distributed " +\
                        "random vector by a vector of different size.")
            else:
                raise ValueError("Cannot multiply Gaussian distributed " +\
                    "random vector by a tensor with more than 1 index.")
        else:
            # assume other is a constant
            new_mean = self.internal_mean.A[0] + other
            new_covariance = self.covariance.A
        return GaussianDistribution(new_mean, new_covariance)
    
    def __radd__(self, other):
        """
        Returns the same thing as __add__ (this makes addition commutative).
        """
        return self.__add__(other)
    
    def __sub__(self, other):
        """
        Returns the same thing as __add__ with an argument of (-other)
        """
        return self.__add__(-other)
    
    def __rsub__(self, other):
        """
        Returns the negative of the returned value of __sub__ (this makes
        subtraction doable from either side).
        """
        return self.__sub__(other).__neg__()
    
    def __neg__(self):
        """
        Returns a new GaussianDistribution with a mean multiplied by -1.
        """
        return GaussianDistribution(-self.internal_mean.A[0],\
            self.covariance.A)
    
    def __mul__(self, other):
        """
        Multiplies the Gaussian random variate described by this distribution
        by the given object.

        other: if other is a constant, the returned Gaussian is the same as
                                       this one with the mean multiplied by
                                       other and the covariance multiplied by
                                       other**2
               if other is a 1D numpy.ndarray, it must be of the same length
                                               as the dimension of this
                                               GaussianDistribution. In this
                                               case, the returned
                                               GaussianDistribution is the
                                               distribution of the dot product
                                               of this Gaussian variate with
                                               other
               if other is a 2D numpy.ndarray, it must have shape
                                               (newparams, self.numparams)
                                               where newparams<=self.numparams
                                               The returned
                                               GaussianDistribution is the
                                               distribution of other (matrix)
                                               multiplied with this Gaussian
                                               variate
        
        returns: GaussianDistribution representing the multiplication of this
                 Gaussian variate by other
        """
        new_mean = self.internal_mean.A[0] * other
        new_covariance = self.covariance.A * (other ** 2)
        return GaussianDistribution(new_mean, new_covariance)
    
    def __rmul__(self, other):
        """
        Returns the same thing as __mul__ (this makes multiplication
        commutative).
        """
        return self.__mul__(other)
    
    def __div__(self, other):
        """
        Returns the same thing as __mul__ with an argument of (1/other).
        """
        return self.__mul__(1 / other)
    
    @property
    def square_root_covariance(self):
        """
        Property storing the square root of the covariance matrix.
        """
        if not hasattr(self, '_square_root_covariance'):
            (eigenvalues, eigenvectors) = npla.eigh(self.covariance.A)
            if np.any(eigenvalues <= 0):
                raise ValueError(("Something went wrong, causing the " +\
                    "square root of the covariance matrix of this " +\
                    "GaussianJumpingDistribution to have at least one " +\
                    "complex element. The eigenvalues of the covariance " +\
                    "matrix are {!s}.").format(eigenvalues))
            eigenvalues = np.sqrt(eigenvalues)
            self._square_root_covariance =\
                np.dot(eigenvectors * eigenvalues[None,:], eigenvectors.T)
        return self._square_root_covariance
    
    def __matmul__(self, other):
        """
        Computes the Kullback-Leibler divergence between this distribution and
        other.
        
        other: a GaussianDistribution object
        
        returns: scalar value of the Kullback-Leibler divergence
        """
        if type(other) in [list, tuple, np.ndarray]:
            other = np.array(other)
            if other.ndim == 1:
                if len(other) == self.numparams:
                    new_mean = np.dot(self.internal_mean.A[0], other)
                    new_covariance =\
                        np.dot(np.dot(self.covariance.A, other), other)
                else:
                    raise ValueError("Cannot multiply Gaussian distributed " +\
                        "random vector by a vector of different size.")
            elif other.ndim == 2:
                if other.shape[1] == self.numparams:
                    if other.shape[0] <= self.numparams:
                        # other is a matrix with self.numparams columns
                        new_mean = np.dot(other, self.internal_mean.A[0])
                        new_covariance =\
                            np.dot(other, np.dot(self.covariance.A, other.T))
                    else:
                        raise ValueError("Cannot multiply Gaussian " +\
                            "distributed random vector by matrix which " +\
                            "will expand the number of parameters because " +\
                            "the covariance matrix of the result would be " +\
                            "singular.")
                else:
                    raise ValueError("Cannot multiply given matrix with " +\
                        "Gaussian distributed random vector because the " +\
                        "axis of its second dimension is not the same " +\
                        "length as the random vector.")
            else:
                raise ValueError("Cannot multiply Gaussian distributed " +\
                    "random vector by a tensor with more than 2 indices.")
        else:
            raise TypeError("Matrix multiplication can only be done with " +\
                "sequence types.")
        return GaussianDistribution(new_mean, new_covariance)
    
    @staticmethod
    def kullback_leibler_divergence(first, second):
        """
        Computes the Kullback-Leibler divergence between two distributions
        represented by GaussianDistribution objects.
        
        first, second: Can be GaussianDistribution objects or 2D numpy.ndarrays
                       representing covariance matrices (if only covariance
                       matrices are given, then the term corresponding to the
                       mean difference is omitted)
        
        returns: the Kullback-Leibler divergence from first to second
        """
        if isinstance(first, GaussianDistribution) and\
            isinstance(second, GaussianDistribution):
            if first.numparams == second.numparams:
                dimension = first.numparams
                delta = first.internal_mean.A[0] - second.internal_mean.A[0]
                sigma_Q_inverse = npla.inv(second.covariance.A)
                sigma_P_times_sigma_Q_inverse =\
                    np.dot(first.covariance.A, sigma_Q_inverse)
                return ((np.sum(np.diag(sigma_P_times_sigma_Q_inverse)) -\
                    dimension -\
                    npla.slogdet(sigma_P_times_sigma_Q_inverse)[1] +\
                    np.dot(delta, np.dot(sigma_Q_inverse, delta))) / 2)
            else:
                raise ValueError("The two given distributions do not have " +\
                    "the same numbers of parameters.")
        elif isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            if first.shape == second.shape:
                if (first.ndim == 2) and (first.shape[0] == first.shape[1]):
                    mean = np.zeros(first.shape[0])
                    first_distribution = GaussianDistribution(mean, first)
                    second_distribution = GaussianDistribution(mean, second)
                    return GaussianDistribution.kullback_leibler_divergence(\
                        first_distribution, second_distribution)
                else:
                    raise ValueError("The covariance matrices given to the " +\
                        "GaussianDistribution class' " +\
                        "kullback_leibler_divergence function were not 2D " +\
                        "square.")
            else:
                raise ValueError("The shapes of the two covariance " +\
                    "matrices given to the GaussianDistribution class' " +\
                    "kullback_leibler_divergence function were not of the " +\
                    "same shape.")
        else:
            raise TypeError("At least one of the distributions given to " +\
                "the kullback_leibler_divergence static method of the " +\
                "GaussianDistribution class was not a GaussianDistribution " +\
                "object.")

    def draw(self, shape=None, random=rand):
        """
        Draws a point from this distribution using numpy.random.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        random: the random number generator to use (default: numpy.random)

        returns a numpy.ndarray containing the values from this draw
        """
        if (self.numparams == 1):
            loc = self.internal_mean.A[0,0]
            scale = np.sqrt(self.covariance.A[0,0])
            return random.normal(loc=loc, scale=scale, size=shape)
        elif type(shape) is type(None):
            return self.internal_mean.A[0] +\
                np.dot(self.square_root_covariance,\
                random.normal(0, 1, size=self.numparams))
        else:
            if type(shape) in int_types:
                shape = (shape,)
            random_vector = random.normal(0, 1, size=shape+(1,self.numparams))
            return self.internal_mean.A +\
                np.sum(random_vector * self.square_root_covariance, axis=-1)

    def log_value(self, point):
        """
        Evaluates the log of the value of this distribution at the given point.
        
        point single number if univariate, numpy.ndarray if multivariate
        
        returns: the log of the value of this distribution at the given point
        """
        if type(point) in numerical_types:
            minus_mean = np.matrix([point]) - self.internal_mean
        elif type(point) in sequence_types:
            minus_mean = np.matrix(point) - self.internal_mean
        else:
            raise ValueError("The type of point provided to a " +\
                "GaussianDistribution was not of a numerical type (should " +\
                "be if distribution is univariate) or of a list type " +\
                "(should be if distribution is multivariate).")
        expon =\
            np.float64(minus_mean * self.inverse_covariance * minus_mean.T) / 2
        return -1. * ((self.log_determinant_covariance / 2) + expon +\
            ((self.numparams * natural_log_two_pi) / 2))

    def to_string(self):
        """
        Finds and returns the string representation of this
        GaussianDistribution.
        """
        if self.numparams == 1:
            return "Normal(mean={0:.3g},variance={1:.3g})".format(\
                self.internal_mean.A[0,0], self.covariance.A[0,0])
        else:
            return "{}-dim Normal".format(len(self.internal_mean.A[0]))
    
    def marginalize(self, key):
        """
        Marginalizes this Gaussian over all of the parameters not described by
        given key.
        
        key: key representing index (indices) to keep. It can be a slice, list,
             numpy.ndarray, or integer.
        
        returns: marginalized GaussianDistribution
        """
        new_mean = self.internal_mean.A[0][key]
        new_covariance = self.covariance.A[:,key][key]
        return GaussianDistribution(new_mean, new_covariance)
    
    def __getitem__(self, key):
        """
        Marginalizes this Gaussian over all of the parameters not described by
        given key.
        
        key: key representing index (indices) to keep. It can be a slice, list,
             numpy.ndarray, or integer.
        
        returns: marginalized GaussianDistribution
        """
        return self.marginalize(key)
    
    def conditionalize(self, known_indices, values):
        """
        Marginalizes this Gaussian over all of the parameters not described by
        given key.
        
        known_indices: key representing index (indices) to keep. It can be a
                       slice, list, numpy.ndarray, or integer.
        values: values of variables corresponding to known_indices
        
        returns: conditionalized GaussianDistribution
        """
        if isinstance(known_indices, slice):
            known_indices = np.arange(*known_indices.indices(self.numparams))
        elif type(known_indices) in int_types:
            known_indices = np.array([known_indices])
        elif type(known_indices) in sequence_types:
            known_indices = np.array(known_indices)
        remaining_indices = np.array([index\
            for index in np.arange(self.numparams)\
            if index not in known_indices])
        new_covariance = npla.inv(\
            self.inverse_covariance.A[:,remaining_indices][remaining_indices])
        known_mean_displacement =\
            values - self.internal_mean.A[0][known_indices]
        new_mean =\
            self.internal_mean.A[0][remaining_indices] -\
            np.dot(new_covariance, np.dot(\
            self.inverse_covariance.A[:,known_indices][remaining_indices],\
            known_mean_displacement))
        return GaussianDistribution(new_mean, new_covariance)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a GaussianDistribution with the same mean (down to 10^-9
        level) and variance (down to 10^-12 dynamic range) and False otherwise.
        """
        if isinstance(other, GaussianDistribution):
            if self.numparams == other.numparams:
                mean_close = np.allclose(self.internal_mean.A,\
                    other.internal_mean.A, rtol=0, atol=1e-9)
                covariance_close = np.allclose(self.covariance.A,\
                    other.covariance.A, rtol=1e-12, atol=0)
                metadata_equal = self.metadata_equal(other)
                return all([mean_close, covariance_close, metadata_equal])
            else:
                return False
        else:
            return False
    
    def inverse_cdf(self, cdf):
        """
        Inverse of the cumulative distribution function. Only expected to make
        sense if numparams == 1
        
        cdf: value between 0 and 1
        """
        return (self.internal_mean.A[0,0] +\
            (np.sqrt(2 * self.covariance.A[0,0]) * erfinv((2 * cdf) - 1)))
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
        """
        return None if (self.numparams == 1) else ([None] * self.numparams)
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        return None if (self.numparams == 1) else ([None] * self.numparams)
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, mean_link=None, covariance_link=None,\
        save_metadata=True):
        """
        Fills the given hdf5 file group with data from this distribution. The
        fact that this is a GaussianDistribution is saved along with the mean
        and covariance.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'GaussianDistribution'
        create_hdf5_dataset(group, 'mean', data=self.internal_mean.A[0],\
            link=mean_link)
        create_hdf5_dataset(group, 'covariance', data=self.covariance.A,\
            link=covariance_link)
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a GaussianDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a GaussianDistribution object created from the information in
                 the given group
        """
        try:
            assert group.attrs['class'] == 'GaussianDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "GaussianDistribution.")
        metadata = Distribution.load_metadata(group)
        mean = get_hdf5_value(group['mean'])
        covariance = get_hdf5_value(group['covariance'])
        return GaussianDistribution(mean, covariance, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivatives of log_value(point) with respect to the
        parameters.
        
        point: either single value (if this Gaussian is 1D) or 1D vector (if
               this Gaussian is ND) at which to evaluate the derivatives
        
        returns: if this Gaussian is 1D, returns single value of derivative
                 else, returns 1D vector of values of derivatives
        """
        if type(point) in numerical_types:
            mean_minus = self.internal_mean - np.matrix([point])
        elif type(point) in sequence_types:
            mean_minus = self.internal_mean - np.matrix(point)
        else:
            raise ValueError("The type of point provided to a " +\
                "GaussianDistribution was not of a numerical type (should " +\
                "be if distribution is univariate) or of a list type " +\
                "(should be if distribution is multivariate).")
        if self.numparams == 1:
            return (mean_minus * self.inverse_covariance).A[0,0]
        else:
            return (mean_minus * self.inverse_covariance).A[0,:]
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivatives of log_value(point) with respect to the
        parameters.
        
        point: vector at which to evaluate the derivatives
        
        returns: 2D square matrix of second derivatives of log value
        """
        if self.numparams == 1:
            return -self.inverse_covariance.A[0,0]
        else:
            return -self.inverse_covariance.A
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return GaussianDistribution(self.internal_mean.A[0].copy(),\
            self.covariance.A.copy())

