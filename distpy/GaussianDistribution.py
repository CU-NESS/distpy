"""
File: distpy/GaussianDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing Gaussian distribution
             (univariate or multivariate).
"""
import numpy as np
import numpy.random as rand
import numpy.linalg as lalg
from .TypeCategories import numerical_types, sequence_types
from .Distribution import Distribution

two_pi = 2 * np.pi

class GaussianDistribution(Distribution):
    """
    A multivariate (or univariate) Gaussian distribution. The classic. Useful
    when some knowledge of the parameters exists and those parameters can be
    any real number.
    """
    def __init__(self, mean, covariance):
        """
        Initializes either a univariate or a multivariate GaussianDistribution.

        mean the mean must be either a number (if univariate)
                                  or a 1D array (if multivariate)
        covariance the covariance must be either a number (if univariate)
                                              or a 2D array (if multivariate)
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
                    self.covariance = np.matrix(arrcov)
                else:
                    raise ValueError("The covariance given to a " +\
                                     "GaussianDistribution was not " +\
                                     "castable to an array of the correct " +\
                                     "shape. It should be a square shape " +\
                                     "with the same side length as length " +\
                                     "of mean.")
            else:
                raise ValueError("The mean of a GaussianDistribution " +\
                                 "is array-like but its covariance" +\
                                 " isn't matrix like.")
        else:
            raise ValueError("The mean of a GaussianDistribution " +\
                             "is not of a recognizable type.")
        self.invcov = lalg.inv(self.covariance)
        self.logdetcov = lalg.slogdet(self.covariance)[1]
    
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
                                 "set to a number but the covariance can't " +\
                                 "be cast into a number.")
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
                                     "random vector by a vector of " +\
                                     "different size.")
            elif other.ndim == 2:
                if other.shape[1] == self.numparams:
                    if other.shape[0] <= self.numparams:
                        # other is a matrix with self.numparams columns
                        new_mean = np.dot(other, self.mean.A[0])
                        new_covariance =\
                            np.dot(other, np.dot(self.covariance.A, other.T))
                    else:
                        raise ValueError("Cannot multiply Gaussian " +\
                                         "distributed random vector by " +\
                                         "matrix which will expand the " +\
                                         "number of parameters because the " +\
                                         "covariance matrix of the result " +\
                                         "would be singular.")
                else:
                    raise ValueError("Cannot multiply given matrix with " +\
                                     "Gaussian distributed random vector " +\
                                     "because the axis of its second " +\
                                     "dimension is not the same length as " +\
                                     "the random vector.")
            else:
                raise ValueError("Cannot multiply Gaussian distributed " +\
                                 "random vector by a tensor with more than " +\
                                 "3 indices.")
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

    def draw(self):
        """
        Draws a point from this distribution using numpy.random.

        returns a numpy.ndarray containing the values from this draw
        """
        if (self.numparams == 1):
            loc = self.mean.A[0,0]
            scale = np.sqrt(self.covariance.A[0,0])
            return rand.normal(loc=loc, scale=scale)
        return rand.multivariate_normal(self.mean.A[0,:], self.covariance.A)

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
                             "GaussianDistribution was not of a numerical " +\
                             "type (should be if distribution is " +\
                             "univariate) or of a list type (should be if " +\
                             "distribution is multivariate).")
        expon = np.float64(minus_mean * self.invcov * minus_mean.T) / 2.
        return -1. * ((self.logdetcov / 2.) + expon +\
            ((self.numparams * np.log(two_pi)) / 2.))

    def to_string(self):
        """
        Finds and returns the string representation of this
        GaussianDistribution.
        """
        if self.numparams == 1:
            return "Normal(mean=%.3g,variance=%.3g)" %\
                (self.mean.A[0,0], self.covariance.A[0,0])
        else:
            return "%i-dim Normal" % (len(self.mean.A[0]),)
    
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
            return mean_close and covariance_close
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this distribution. The
        fact that this is a GaussianDistribution is saved along with the mean
        and covariance.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'GaussianDistribution'
        group.create_dataset('mean', data=self.mean.A[0])
        group.create_dataset('covariance', data=self.covariance.A)

