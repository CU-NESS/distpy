"""
File: distpy/distribution/GaussianDistribution.py
Author: Keith Tauscher
Date: Feb 12 2018

Description: File containing class representing Gaussian distribution
             (univariate or multivariate).
"""
import numpy as np
import numpy.random as rand
import numpy.linalg as npla
import scipy.linalg as scila
from scipy.special import erfinv
from ..util import numerical_types, int_types, sequence_types,\
    create_hdf5_dataset, get_hdf5_value
from .Distribution import Distribution

two_pi = 2 * np.pi

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
        if type(mean) in numerical_types:
            self._check_covariance_when_mean_has_size_1(mean,\
                                                        covariance)
        elif type(mean) in sequence_types:
            arrmean = np.array(mean)
            if arrmean.ndim != 1:
                raise ValueError("The mean of a GaussianDistribution was " +\
                    "not 1 dimensional.")
            elif arrmean.size == 0:
                raise ValueError("The mean of a GaussianDistribution was " +\
                    "set to something like an empty array.")
            elif arrmean.size == 1:
                self._check_covariance_when_mean_has_size_1(mean[0],\
                                                            covariance)
            elif type(covariance) in sequence_types:
                arrcov = np.array(covariance)
                if arrcov.shape == (len(arrmean), len(arrmean)):
                    self.mean = np.matrix(arrmean)
                    self._numparams = len(arrmean)
                    self.covariance = np.matrix((arrcov + arrcov.T) / 2.)
                else:
                    raise ValueError("The covariance given to a " +\
                        "GaussianDistribution was not castable to an array " +\
                        "of the correct shape. It should be a square shape " +\
                        "with the same side length as length of mean.")
            else:
                raise ValueError("The mean of a GaussianDistribution is " +\
                    "array-like but its covariance isn't matrix like.")
        else:
            raise ValueError("The mean of a GaussianDistribution is not of " +\
                "a recognizable type.")
        self.invcov = npla.inv(self.covariance)
        self.logdetcov = npla.slogdet(self.covariance)[1]
        self.metadata = metadata
    
    def _check_covariance_when_mean_has_size_1(self, true_mean, covariance):
        #
        # If the mean is a single number, then the covariance should be
        # castable into a single number as well. This function checks that and
        # raises an error if something unexpected happens. This function sets
        # self.mean and self.covariance.
        #
        # true_mean the single number mean (should be a numerical_type)
        # covariance the covariance which *should* be castable into a number
        #
        if type(covariance) in numerical_types:
            # covariance is single number, as it should be
            self.covariance = np.matrix([[covariance]])
        elif type(covariance) in sequence_types:
            # covariance should be number but at first glance, it isn't
            arrcov = np.array(covariance)
            if arrcov.size == 1:
                self.covariance = np.matrix([[arrcov[(0,) * arrcov.ndim]]])
            else:
                raise ValueError("The mean of a GaussianDistribution was " +\
                    "set to a number but the covariance can't be cast into " +\
                    "a number.")
        else:
            raise ValueError("The covariance of a GaussianDistribution is " +\
                "not of a recognizable type.")
        self.mean = np.matrix([true_mean])
        self._numparams = 1

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters which this
        GaussianDistribution describes (same as dimension of mean and
        covariance).
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("For some reason, I don't know how many " +\
                "parameters this GaussianDistribution has!")
        return self._numparams
    
    def __add__(self, other):
        """
        Adds other to this Gaussian variate. The result of this operation is a
        Gaussian with a shifted mean but identical covariance.
        
        other: must be castable to the 1D array shape of the Gaussian variate
               described by this distribution
        """
        return GaussianDistribution(self.mean.A[0] + other, self.covariance.A)
    
    def __radd__(self, other):
        """
        Returns the same thing as __add__ (this makes addition commutative).
        """
        return self.__add__(other)
    
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
        if type(other) in [list, tuple, np.ndarray]:
            other = np.array(other)
            if other.ndim == 1:
                if len(other) == self.numparams:
                    new_mean = np.dot(self.mean.A[0], other)
                    new_covariance =\
                        np.dot(np.dot(self.covariance.A, other), other)
                else:
                    raise ValueError("Cannot multiply Gaussian distributed " +\
                        "random vector by a vector of different size.")
            elif other.ndim == 2:
                if other.shape[1] == self.numparams:
                    if other.shape[0] <= self.numparams:
                        # other is a matrix with self.numparams columns
                        new_mean = np.dot(other, self.mean.A[0])
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
                    "random vector by a tensor with more than 3 indices.")
        else:
            # assume other is a constant
            new_mean = self.mean.A[0] * other
            new_covariance = self.covariance.A * (other ** 2)
        return GaussianDistribution(new_mean, new_covariance)
        
    
    def __rmul__(self, other):
        """
        Returns the same thing as __mul__ (this makes multiplication
        commutative).
        """
        return self.__mul__(other)
    
    @property
    def square_root_covariance(self):
        """
        Property storing the square root of the covariance matrix.
        """
        if not hasattr(self, '_square_root_covariance'):
            self._square_root_covariance = scila.sqrtm(self.covariance)
        return self._square_root_covariance

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
            loc = self.mean.A[0,0]
            scale = np.sqrt(self.covariance.A[0,0])
            return random.normal(loc=loc, scale=scale, size=shape)
        elif shape is None:
            return self.mean.A[0] + np.dot(self.square_root_covariance,\
                random.normal(0, 1, size=self.numparams))
        else:
            if type(shape) in int_types:
                shape = (shape,)
            random_vector = random.normal(0, 1, size=shape+(1,self.numparams))
            return self.mean.A +\
                np.sum(random_vector * self.square_root_covariance, axis=-1)

    def log_value(self, point):
        """
        Evaluates the log of the value of this distribution at the given point.
        
        point single number if univariate, numpy.ndarray if multivariate
        
        returns: the log of the value of this distribution at the given point
        """
        if type(point) in numerical_types:
            minus_mean = np.matrix([point]) - self.mean
        elif type(point) in sequence_types:
            minus_mean = np.matrix(point) - self.mean
        else:
            raise ValueError("The type of point provided to a " +\
                "GaussianDistribution was not of a numerical type (should " +\
                "be if distribution is univariate) or of a list type " +\
                "(should be if distribution is multivariate).")
        expon = np.float64(minus_mean * self.invcov * minus_mean.T) / 2.
        return -1. * ((self.logdetcov / 2.) + expon +\
            ((self.numparams * np.log(two_pi)) / 2.))

    def to_string(self):
        """
        Finds and returns the string representation of this
        GaussianDistribution.
        """
        if self.numparams == 1:
            return "Normal(mean={0:.3g},variance={1:.3g})".format(\
                self.mean.A[0,0], self.covariance.A[0,0])
        else:
            return "{}-dim Normal".format(len(self.mean.A[0]))
    
    def marginalize(self, key):
        """
        Marginalizes this Gaussian over all of the parameters not described by
        given key.
        
        key: key representing index (indices) to keep. It can be a slice, list,
             numpy.ndarray, or integer.
        
        returns: marginalized GaussianDistribution
        """
        new_mean = self.mean.A[0][key]
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
        elif isinstance(known_indices, int):
            known_indices = np.array([known_indices])
        elif type(known_indices) in sequence_types:
            known_indices = np.array(known_indices)
        remaining_indices = np.array([index\
            for index in np.arange(self.numparams)\
            if index not in known_indices])
        new_covariance =\
            npla.inv(self.invcov.A[:,remaining_indices][remaining_indices])
        known_mean_displacement = values - self.mean.A[0][known_indices]
        new_mean = self.mean.A[0][remaining_indices] - np.dot(new_covariance,\
            np.dot(self.invcov.A[:,known_indices][remaining_indices],\
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
                mean_close =\
                    np.allclose(self.mean.A, other.mean.A, rtol=0, atol=1e-9)
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
        return (self.mean.A[0,0] +\
            (np.sqrt(2 * self.covariance.A[0,0]) * erfinv((2 * cdf) - 1)))
    
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
        create_hdf5_dataset(group, 'mean', data=self.mean.A[0], link=mean_link)
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
            mean_minus = self.mean - np.matrix([point])
        elif type(point) in sequence_types:
            mean_minus = self.mean - np.matrix(point)
        else:
            raise ValueError("The type of point provided to a " +\
                "GaussianDistribution was not of a numerical type (should " +\
                "be if distribution is univariate) or of a list type " +\
                "(should be if distribution is multivariate).")
        if self.numparams == 1:
            return (mean_minus * self.invcov).A[0,0]
        else:
            return (mean_minus * self.invcov).A[0,:]
    
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
            return -self.invcov.A[0,0]
        else:
            return -self.invcov.A

