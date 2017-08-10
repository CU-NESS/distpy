"""
File: distpy/GaussianDirectionDistribution.py
Author: Keith Tauscher
Date: 10 Aug 2017

Description: File containing class representing Gaussian distribution on the
             surface of the sphere.
"""
import numpy as np
import numpy.random as rand
import healpy as hp
from .TypeCategories import int_types
from .Distribution import Distribution
from .UniformDistribution import UniformDistribution

class GaussianDirectionDistribution(Distribution):
    """
    Class representing Gaussian distribution on the surface of the sphere.
    """
    def __init__(self, pointing_center=(90, 0), sigma=1, degrees=True):
        """
        Generates a new GaussianPointingPrior centered at the given pointing
        and with the given angular scale.
        
        pointing_center is always given in degrees, no matter value of degrees
        sigma angular scale of Gaussian
        degrees: if True, sigma is in degres; if False, sigma is in radians
        """
        self.pointing_center = pointing_center
        self.theta_center = 90 - pointing_center[0]
        self.phi_center = pointing_center[1]
        self.sigma = sigma
        if degrees:
            self.sigma = np.radians(self.sigma)
        self.psi_prior = UniformDistribution(0, 2 * np.pi)
        rot_zprime = hp.rotator.Rotator(rot=(-self.phi_center, 0, 0),\
            deg=True, eulertype='y')
        rot_yprime = hp.rotator.Rotator(rot=(0, self.theta_center, 0),\
            deg=True, eulertype='y')
        self.rotator = rot_zprime * rot_yprime
        self.log_pdf_constant = -np.log((self.sigma ** 2) * 2 * np.pi)
    
    @property
    def numparams(self):
        """
        number of parameters is always 1 for pointing properties because they
        specify a single point on the celestial sphere.
        """
        return 2
    
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
        gamma = hp.rotator.angdist((point[1], point[0]),\
            (self.pointing_center[1], self.pointing_center[0]), lonlat=True)[0]
        return self.log_pdf_constant - (((gamma / self.sigma) ** 2) / 2)
    
    def draw(self, shape=None):
        """
        Draws value(s) from this distribution.
        
        shape: if None, returns single pair (latitude, longitude) in degrees
               if int, n, returns n random variates (array of shape (n, 2))
               if tuple of n ints, (n+1)-D array
        """
        psi_draw = self.psi_prior.draw(shape=shape)
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
        group.attrs['pointing_center'] = self.pointing_center
        group.attrs['sigma'] = self.sigma

