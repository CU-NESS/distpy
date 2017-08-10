"""
File: distpy/UniformDirectionDistribution.py
Author: Keith Tauscher
Date: 9 Aug 2017

Description: File containing class representing uniform distribution on the
             (2D) surface of the sphere.
"""
import numpy as np
import healpy as hp
from .TypeCategories import int_types
from .Distribution import Distribution
from .UniformDistribution import UniformDistribution

class UniformDirectionDistribution(Distribution):
    """
    Class representing a uniform distribution on the surface of a sphere.
    """
    def __init__(self, low_theta=0, high_theta=np.pi, low_phi=0,\
        high_phi=2*np.pi, pointing_center=(90, 0), psi_center=0):
        """
        low_theta, high_theta, low_phi, and high_phi are given in radians.
        pointing_center is given in (lat, lon) in degrees and psi_center is
        given in degrees
        """
        self.low_theta = low_theta
        self.high_theta = high_theta
        self.low_phi = low_phi
        self.high_phi = high_phi
        self.psi_center = psi_center
        self.pointing_center = pointing_center
        self.theta_center = 90 - self.pointing_center[0]
        self.phi_center = self.pointing_center[1]
        self.cos_low_theta = np.cos(self.low_theta)
        self.cos_high_theta = np.cos(self.high_theta)
        self.phi_distribution =\
            UniformDistribution(self.low_phi, self.high_phi)
        self.cos_theta_distribution =\
            UniformDistribution(self.cos_high_theta, self.cos_low_theta)
        self.delta_cos_theta = self.cos_low_theta - self.cos_high_theta
        self.delta_phi = self.high_phi - self.low_phi
        self.delta_omega = self.delta_cos_theta * self.delta_phi
        self.const_log_value = -np.log(self.delta_omega)
        rot_zprime = hp.rotator.Rotator(rot=(-self.phi_center, 0, 0),\
            deg=True, eulertype='y')
        rot_yprime = hp.rotator.Rotator(rot=(0, self.theta_center, 0),\
            deg=True, eulertype='y')
        rot_z = hp.rotator.Rotator(rot=(self.psi_center, 0, 0), deg=True,\
            eulertype='y')
        self.rotator = rot_zprime * rot_yprime * rot_z

    def draw(self, shape=None):
        """
        Draws a direction from this distribution.
        
        shape: if None, returns single pair (latitude, longitude) in degrees
               if int, n, returns n random variates (array of shape (n, 2))
               if tuple of n ints, (n+1)-D array
        """
        if shape is None:
            phi_draw = self.phi_distribution.draw()
            theta_draw = np.arccos(self.cos_theta_distribution.draw())
        else:
            if type(shape) in int_types:
                shape = (shape,)
            phi_draw = self.phi_distribution.draw(shape=shape).flatten()
            theta_draw = np.arccos(self.cos_theta_distribution.draw(\
                shape=shape).flatten())
        (theta, phi) = self.rotator(theta_draw, phi_draw)
        if shape is not None:
            (theta, phi) = (np.reshape(theta, shape), np.reshape(phi, shape))
        return np.stack([90 - np.degrees(theta), np.degrees(phi)], axis=-1)
    
    def log_value(self, point):
        """
        Calculates the log of the value of this distribution at given point.
        
        point: length-2 sequence containing (latitude, longitude) in degrees
        
        returns: natural logarithm of value of this distribution at point
        """
        rotated = self.rotator.I(point[1], point[0], lonlat=True)
        theta = np.radians(90 - rotated[1])
        phi = np.radians(rotated[0] % 360.)
        if (theta < self.low_theta) or (theta > self.high_theta) or\
            (phi < self.low_phi) or (phi > self.high_phi):
            return -np.inf
        return self.const_log_value
    
    def to_string(self):
        """
        Returns a string representation of this distribution.
        """
        return "UniformDirection((%.3g, %.3g), %.3g, %.3g, %.3g, %.3g)" %\
            (self.pointing_center[0], self.pointing_center[1], self.low_theta,\
            self.high_theta, self.low_phi, self.high_phi)
    
    @property
    def numparams(self):
        """
        pointing directions are 2D because the surface of the sphere is 2D
        """
        return 2
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True iff
        other is a UniformDirectionDistribution with the same bounds.
        """
        if isinstance(other, UniformDirectionDistribution):
            these_properties = [self.low_theta, self.high_theta, self.low_phi,\
                self.high_phi]
            other_properties = [other.low_theta, other.high_theta,\
                other.low_phi, other.high_phi]
            return np.allclose(these_properties, other_properties, rtol=0,\
                atol=1e-9)
        else:
            return False

    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with data about this distribution.
        
        group: hdf5 file group to fill with data about this distribution
        """
        group.attrs['class'] = 'UniformDirectionDistribution'
        group.attrs['low_theta'] = self.low_theta
        group.attrs['high_theta'] = self.high_theta
        group.attrs['low_phi'] = self.low_phi
        group.attrs['high_phi'] = self.high_phi
        group.attrs['pointing_center'] = self.pointing_center
        group.attrs['psi_center'] = self.psi_center

