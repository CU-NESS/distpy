"""
File: distpy/jumping/GaussianJumpingDistribution.py
Author: Keith Tauscher
Date: 21 Dec 2017

Description: File containing a jumping distribution which is a uniform ellipse
             centered on the source point and defined by a given covariance.
"""
import numpy as np
import numpy.linalg as npla
import scipy.linalg as scila
from scipy.special import gammaln as log_gamma
from ..util import int_types, numerical_types, sequence_types,\
    create_hdf5_dataset
from .JumpingDistribution import JumpingDistribution

class UniformJumpingDistribution(JumpingDistribution):
    """
    Class representing a jumping distribution which is centered on the source
    point and has the given covariance.
    """
    def __init__(self, covariance):
        """
        Initializes a UniformJumpingDistribution with the given covariance
        matrix.
        
        covariance: either single number (if this should be a 1D uniform) or
                    square 2D array (if this should be a multivariate ellipse)
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
            if (value.ndim == 2) and (value.shape[0] == value.shape[1]):
                self._covariance = value
            else:
                raise ValueError("covariance didn't have the expected shape.")
        else:
            raise TypeError("covariance was neither a number nor an array.")
        self.inverse_covariance, self.constant_log_value # compute stuff
    
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
    def constant_log_value(self):
        """
        Property storing a constant in the log value which is independent of
        both the source and the destination.
        """
        if not hasattr(self, '_constant_log_value'):
            n_over_2 = self.numparams / 2.
            n_plus_2 = self.numparams + 2
            self._constant_log_value = log_gamma(n_over_2 + 1) -\
                (n_over_2 * (np.log(np.pi * (n_plus_2)))) -\
                (npla.slogdet(self.covariance)[1] / 2.)
        return self._constant_log_value
    
    @property
    def matrix_for_draw(self):
        """
        Property storing the matrix square root of
        (self.covariance * (self.numparams + 2)), which plays an important role
        in the drawing from this JumpingDistribution.
        """
        return scila.sqrtm(self.covariance * (self.numparams + 2))
    
    def draw(self, source, shape=None):
        """
        Draws a destination point from this jumping distribution given a source
        point.
        
        source: if this JumpingDistribution is univariate, source should be a
                                                           single number
                otherwise, source should be numpy.ndarray of shape (numparams,)
        
        returns: destination in same format as source
        """
        if self.numparams == 1:
            return np.random.uniform(source - self.half_span,\
                source + self.half_span, size=shape)
        else:
            none_shape = (shape is None)
            if none_shape:
                shape = (1,)
            elif type(shape) in int_types:
                shape = (shape,)
            normal_vector =\
                np.random.standard_normal(size=shape+(self.numparams,))
            radii = np.power(np.random.random(size=shape), 1. / self.numparams)
            radii = (radii / npla.norm(normal_vector, axis=-1))[...,np.newaxis]
            displacement = radii * np.dot(normal_vector, self.matrix_for_draw)
            destination = displacement +\
                source[((np.newaxis,)*len(shape))+(slice(None),)]
            if none_shape:
                return destination[0]
            else:
                return destination
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF: ln(f(source->destination))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        difference = (destination - source)
        chi2 = np.dot(difference, np.dot(difference, self.inverse_covariance))
        if chi2 < (self.numparams + 2):
            return self.constant_log_value
        else:
            return -np.inf
    
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
    def half_span(self):
        """
        """
        if not hasattr(self, '_half_span'):
            if self.numparams == 1:
                self._half_span = np.sqrt(self.covariance[0,0] * 3)
            else:
                raise NotImplementedError("The span of a multivariate " +\
                    "distribution is not well-defined and thus can't be " +\
                    "referenced.")
        return self._half_span
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: JumpingDistribution with which to check for equality
        
        returns: True or False
        """
        if isinstance(other, UniformJumpingDistribution):
            if self.numparams == other.numparams:
                return np.allclose(self.covariance, other.covariance,\
                    rtol=1e-12, atol=1e-12)
            else:
                return False
        else:
            return False
    
    def fill_hdf5_group(self, group, covariance_link=None):
        """
        Fills the given hdf5 file group with data from this distribution. The
        fact that this is a GaussianJumpingDistribution is saved along with the
        mean and covariance.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'UniformJumpingDistribution'
        create_hdf5_dataset(group, 'covariance', data=self.covariance,\
            link=covariance_link)

