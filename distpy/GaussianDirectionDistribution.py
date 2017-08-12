"""
File: distpy/GaussianDirectionDistribution.py
Author: Keith Tauscher
Date: 10 Aug 2017

Description: File containing class representing Gaussian distribution on the
             surface of the sphere.
"""
import numpy as np
import numpy.random as rand
from .TypeCategories import int_types, numerical_types
from .DirectionDistribution import DirectionDistribution
from .UniformDistribution import UniformDistribution

class GaussianDirectionDistribution(DirectionDistribution):
    """
    Class representing Gaussian distribution on the surface of the sphere.
    """
    def __init__(self, pointing_center=(90, 0), sigma=1, degrees=True):
        """
        Generates a new GaussianPointingPrior centered at the given pointing
        and with the given angular scale.
        
        pointing_center: (latitude, longitude) always given in degrees, no
                         matter value of degrees parameter. Only (90,0) is
                         allowed if healpy is not installed.
        sigma: angular scale of Gaussian. in degrees if degrees parameter is
               True
        degrees: if True, sigma is in degres; if False, sigma is in radians
        """
        self.psi_center = 0
        self.pointing_center = pointing_center
        if degrees:
            self.sigma = np.radians(sigma)
        else:
            self.sigma = sigma
    
    @property
    def psi_distribution(self):
        """
        Property storing the distribution of the azimuthal angle about
        pointing_center.
        """
        if not hasattr(self, '_psi_distribution'):
            self._psi_distribution = UniformDistribution(0, 2 * np.pi)
        return self._psi_distribution
    
    @property
    def sigma(self):
        """
        Property storing the angular scale of this distribution in radians.
        """
        if not hasattr(self, '_sigma'):
            raise AttributeError("sigma was referenced before it was set.")
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        """
        Setter for the angular scale of this distribution (in radians).
        
        value: single positive number (in radians)
        """
        if type(sigma) in numerical_types:
            if sigma > 0:
                self._sigma = value
            else:
                raise ValueError("sigma given to " +\
                                 "GaussianDirectionDistribution was not " +\
                                 "positive.")
        else:
            raise TypeError("sigma given to GaussianDirectionDistribution " +\
                            "was not a single number.")
    
    @property
    def const_log_value_contribution(self):
        """
        Property storing the constant part of the logarithm of the value of the
        distribution at each given point.
        """
        if not hasattr(self, '_const_log_value_contribution'):
            self._const_log_value_contribution =\
                -np.log((self.sigma ** 2) * 2 * np.pi)
        return self._const_log_value_contribution
    
    def to_string(self):
        """
        Creates a string representation of this distribution.
        """
        return 'GaussianDirection((%.3g, %.3g), %.3g)' %\
            (self.pointing_center[0], self.pointing_center[1], self.sigma)
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at point.
        
        point: tuple of form (latitude, longitude)
        """
        sine_latitude_product = np.sin(np.radians(point[0])) *\
            self.cos_theta_center
        cosine_latitude_product = np.cos(np.radians(point[0])) *\
            self.sin_theta_center
        cosine_longitude_difference =\
            np.cos(np.radians(point[1] - self.phi_center))
        gamma = np.arccos(sine_latitude_product +\
            (cosine_latitude_product * cosine_longitude_difference))
        return self.const_log_value_contribution -\
            (((gamma / self.sigma) ** 2) / 2)
    
    def draw(self, shape=None):
        """
        Draws value(s) from this distribution.
        
        shape: if None, returns single pair (latitude, longitude) in degrees
               if int, n, returns n random variates (array of shape (n, 2))
               if tuple of n ints, (n+1)-D array
        """
        psi_draw = self.psi_distribution.draw(shape=shape)
        if shape is None:
            gamma_draw =\
                self.sigma * np.sqrt(-2 * np.log(1 - rand.rand()))
        else:
            if type(shape) in int_types:
                shape = (shape,)
            gamma_draw =\
                self.sigma * np.sqrt(-2 * np.log(1 - rand.rand(*shape)))
        theta_draw, phi_draw = self.rotator(gamma_draw, psi_draw)
        return np.stack([90 - np.degrees(theta_draw), np.degrees(phi_draw)],\
            axis=-1)
    
    def __eq__(self, other):
        """
        Checks for equality between this and other. Returns True iff
        theta_center, phi_center, and sigma are all equal.
        """
        if isinstance(other, GaussianDirectionDistribution):
            these_properties = [self.theta_center, self.phi_center, self.sigma]
            other_properties =\
                [other.theta_center, other.phi_center, other.sigma]
            return np.allclose(these_properties, other_properties, rtol=0,\
                atol=1e-9)
        else:
            return False

    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this distribution.
        
        group: hdf5 file group to file with data about this distribution
        """
        group.attrs['class'] = 'GaussianDirectionDistribution'
        DirectionDistribution.fill_hdf5_group(self, group)
        group.attrs['sigma'] = self.sigma
