"""
File: distpy/EllipticalUniformDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing an elliptical uniform
             multivariate distribution.
"""
import numpy as np
import numpy.random as rand
import numpy.linalg as lalg
import scipy.linalg as slalg
from scipy.special import gammaln as log_gamma
from ..util import int_types
from .Distribution import Distribution

class EllipticalUniformDistribution(Distribution):
    """
    Distribution on a set of variables where the variables are equally likely
    to be at any point within an ellipsoid (defined by mean and cov). It is a
    uniform distribution over an arbitrary ellipsoid.
    """
    def __init__(self, mean, cov):
        """
        Initializes this EllipticalUniformDistribution using properties of the
        ellipsoid defining it.
        
        mean the center of the ellipse defining this distribution
        cov the covariance describing the ellipse defining this distribution. A
            consequence of this definition is that, in order for a point, x, to
            be in the ellipse, (x-mean)^T*cov^-1*(x-mean) < N+2 must be
            satisfied
        """
        try:
            self.mean = np.array(mean)
        except:
            raise TypeError("mean given to EllipticalUniformDistribution " +\
                            "could not be cast as a numpy.ndarray.")
        try:
            self.cov = np.array(cov)
        except:
            raise TypeError("cov given to EllipticalUniformDistribution " +\
                            "could not be cast as a numpy.ndarray.")
        if (self.cov.shape != (2 * self.mean.shape)) or (self.mean.ndim != 1):
            raise ValueError("The shapes of the mean and cov given to " +\
                             "EllipticalUniformDistribution did not make " +\
                             "sense. They should fit the following " +\
                             "pattern: mean.shape=(rank,) and " +\
                             "cov.shape=(rank,rank).")
        self._numparams = self.mean.shape[0]
        if self.numparams < 2:
            raise NotImplementedError("The EllipticalUniformDistribution " +\
                                      "doesn't take single variable random " +\
                                      "variates since, in the 1D case, it " +\
                                      "is the same as a simple uniform " +\
                                      "distribution; so, using the " +\
                                      "EllipticalUniformDistribution class " +\
                                      "would involve far too much " +\
                                      "computational overhead.")
        half_rank = self.numparams / 2.
        self.invcov = lalg.inv(self.cov)
        self.const_log_value = log_gamma(half_rank + 1) -\
            (half_rank * (np.log(np.pi) + np.log(self.numparams + 2))) -\
            (lalg.slogdet(self.cov)[1] / 2.)
        self.sqrtcov = slalg.sqrtm(self.cov)

    @property
    def numparams(self):
        """
        The number of parameters which are represented in this distribution.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.mean)
        return self._numparams

    def draw(self, shape=None):
        """
        Draws a random vector from this uniform elliptical distribution. By the
        definition of this class, the point it draws is equally likely to lie
        anywhere inside the ellipse defining this distribution.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        
        returns numpy.ndarray of containing random variates for each parameter
        """
        none_shape = (shape is None)
        if none_shape:
            shape = (1,)
        elif type(shape) in int_types:
            shape = (shape,)
        xis = rand.randn(*(shape + (self.numparams,)))
        xis = xis / np.sqrt(np.sum(np.power(xis, 2), axis=-1, keepdims=True))
        # xi now contains random directional unit vectors
        radial_cdfs = rand.rand(*shape)
        max_z_radius = np.sqrt(self.numparams + 2)
        fractional_radii = np.power(radial_cdfs, 1. / self.numparams)
        deviations = max_z_radius * fractional_radii[...,np.newaxis] *\
            np.dot(xis, self.sqrtcov)
        points = self.mean[((np.newaxis,)*len(shape)) + (slice(None),)] +\
            deviations
        if none_shape:
            return points[0]
        else:
            return points
    
    def log_value(self, point):
        """
        Evaluates the log of this distribution at the given point.
        
        point: the vector point of parameters at which to calculate the
               numerical point of this distribution
        
        returns: if point is inside ellipse, ln(V) where V is volume of
                                             ellipsoid
                 if point is outside ellipse, -np.inf
        """
        centered_point = np.array(point) - self.mean
        matprod = np.dot(np.dot(centered_point, self.invcov), centered_point)
        if (matprod <= (self.numparams + 2)):
            return self.const_log_value
        else:
            return -np.inf

    def to_string(self):
        """
        Gives a simple string (of the form: "N-dim elliptical" where N is the
        number of parameters) summary of this distribution.
        """
        return ('%i-dim elliptical' % (self.numparams,))
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is an EllipticalUniformDistribution of the same dimension with
        the same mean (down to 10^-9 level) and covariance (down to dynamic
        range of 10^-12) and False otherwise.
        """
        if isinstance(other, EllipticalUniformDistribution):
            if self.numparams == other.numparams:
                mean_close =\
                    np.allclose(self.mean, other.mean, rtol=0, atol=1e-9)
                cov_close = np.allclose(self.cov, other.cov, rtol=1e-12, atol=0)
                return mean_close and cov_close
            else:
                return False
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this distribution. The
        data to be saved includes the class name, mean, and covariance of this
        distribution.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'EllipticalUniformDistribution'
        group.create_dataset('mean', data=self.mean)
        group.create_dataset('covariance', data=self.cov)

