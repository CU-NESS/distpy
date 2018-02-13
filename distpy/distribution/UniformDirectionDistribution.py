"""
File: distpy/UniformDirectionDistribution.py
Author: Keith Tauscher
Date: 9 Aug 2017

Description: File containing class representing uniform distribution on the
             (2D) surface of the sphere.
"""
import numpy as np
from ..util import int_types, numerical_types
from .DirectionDistribution import DirectionDistribution
from .UniformDistribution import UniformDistribution

class UniformDirectionDistribution(DirectionDistribution):
    """
    Class representing a uniform distribution on the surface of a sphere.
    """
    def __init__(self, low_theta=0, high_theta=np.pi, low_phi=0,\
        high_phi=2*np.pi, pointing_center=(90, 0), psi_center=0,\
        metadata=None):
        """
        Initializes a new UniformDirectionDistribution with the given
        parameters.
        
        low_theta, high_theta, low_phi, and high_phi: constants defining
                                                      support of distribution.
                                                      theta (phi) represents
                                                      the polar (azimuthal)
                                                      angle are given in
                                                      radians.
        pointing_center: (latitude, longitude) in degrees. Must be (90, 0)
                         (i.e. left default) if healpy is not installed.
        psi_center: single number in degrees. Must be 0 (i.e. left default) if
                    healpy is not installed
        """
        self.psi_center = psi_center
        self.pointing_center = pointing_center
        self.low_theta = low_theta
        self.high_theta = high_theta
        self.low_phi = low_phi
        self.high_phi = high_phi
        self.metadata = metadata
    
    @property
    def low_theta(self):
        """
        Property storing the lowest value of the polar angle of distribution
        support in radians.
        """
        if not hasattr(self, '_low_theta'):
            raise AttributeError("low_theta was referenced before it was set.")
        return self._low_theta
    
    @low_theta.setter
    def low_theta(self, value):
        """
        Setter for the lowest value of the polar angle of distribution support in
        radians.
        
        value: Must be between 0 and pi, inclusive.
        """
        if type(value) in numerical_types:
            if (value >= 0) and (value <= np.pi):
                self._low_theta = value
            else:
                raise ValueError("low_theta must be between 0 and pi.")
        else:
            raise TypeError("low_theta must be single number.")
    
    @property
    def cos_low_theta(self):
        """
        Property storing cosine of smallest polar angle in the support of this
        distribution.
        """
        if not hasattr(self, '_cos_low_theta'):
            self._cos_low_theta = np.cos(self.low_theta)
        return self._cos_low_theta
    
    @property
    def high_theta(self):
        """
        Property storing the highest value of the polar angle of distribution
        support in radians.
        """
        if not hasattr(self, '_high_theta'):
            raise AttributeError("high_theta was referenced before it was " +\
                "set.")
        return self._high_theta
    
    @high_theta.setter
    def high_theta(self, value):
        """
        Setter for the highest value of the polar angle of distribution support
        in radians.
        
        value: Must be between 0 and pi, inclusive.
        """
        if type(value) in numerical_types:
            if (value >= self.low_theta) and (value <= np.pi):
                self._high_theta = value
            else:
                raise ValueError("high_theta must be between low_theta and " +\
                    "pi.")
        else:
            raise TypeError("high_theta must be single number.")
    
    @property
    def cos_high_theta(self):
        """
        Property storing the cosine of the largest polar angle in the support
        of this distribution.
        """
        if not hasattr(self, '_cos_high_theta'):
            self._cos_high_theta = np.cos(self.high_theta)
        return self._cos_high_theta
    
    @property
    def delta_cos_theta(self):
        """
        Property storing the difference between the largest and smallest values
        of the cosine of the polar angle.
        """
        if not hasattr(self, '_delta_cos_theta'):
            self._delta_cos_theta = self.cos_low_theta - self.cos_high_theta
        return self._delta_cos_theta
    
    @property
    def cos_theta_distribution(self):
        """
        Property storing the distribution of the cosine of the polar angle.
        """
        if not hasattr(self, '_cos_theta_distribution'):
            self._cos_theta_distribution =\
                UniformDistribution(self.cos_high_theta, self.cos_low_theta)
        return self._cos_theta_distribution
    
    @property
    def low_phi(self):
        """
        Property storing the lowest value of the azimuthal angle of
        distribution support in radians.
        """
        if not hasattr(self, '_low_phi'):
            raise AttributeError("low_phi was referenced before it was set.")
        return self._low_phi
    
    @low_phi.setter
    def low_phi(self, value):
        """
        Setter for the lowest value of the azimuthal angle of distribution
        support in radians.
        
        value: single number
        """
        if type(value) in numerical_types:
            self._low_phi = value
        else:
            raise TypeError("low_phi must be single number.")
    
    @property
    def high_phi(self):
        """
        Property storing the highest value of the azimuthal angle of
        distribution support in radians.
        """
        if not hasattr(self, '_high_phi'):
            raise AttributeError("high_phi was referenced before it was set.")
        return self._high_phi
    
    @high_phi.setter
    def high_phi(self, value):
        """
        Setter for the highest value of the azimuthal angle of distribution
        support in radians.
        
        value: single number larger than low_phi
        """
        if type(value) in numerical_types:
            if value > self.low_phi:
                self._high_phi = value
            else:
                raise ValueError("high_phi must be greater than low_phi.")
        else:
            raise TypeError("high_phi must be single number.")
    
    @property
    def delta_phi(self):
        """
        Property storing difference between largest and smallest values of
        azimuthal angle in distribution.
        """
        if not hasattr(self, '_delta_phi'):
            self._delta_phi = self.high_phi - self.low_phi
        return self._delta_phi
    
    @property
    def phi_distribution(self):
        """
        Property storing distribution of the azimuthal angle in radians.
        """
        if not hasattr(self, '_phi_distribution'):
            self._phi_distribution =\
                UniformDistribution(self.low_phi, self.high_phi)
        return self._phi_distribution
    
    @property
    def delta_omega(self):
        """
        Property storing the size (in solid angle) of the support of this
        distribution. The value of the distribution is 1 / delta_omega.
        """
        if not hasattr(self, '_delta_omega'):
            self._delta_omega = self.delta_cos_theta * self.delta_phi
        return self._delta_omega
    
    @property
    def const_log_value(self):
        """
        Property storing the logarithm of the constant value of the
        distribution of any point inside the support of the distribution.
        
        returns: -np.log(self.delta_omega)
        """
        if not hasattr(self, '_const_log_value'):
            self._const_log_value = -np.log(self.delta_omega)
        return self._const_log_value

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
        else:
            return self.const_log_value
    
    def to_string(self):
        """
        Returns a string representation of this distribution.
        """
        return ("UniformDirection(({0:.3g}, {1:.3g}), {2:.3g}, {3:.3g}, " +\
            "{4:.3g}, {5:.3g})").format(self.pointing_center[0],\
            self.pointing_center[1], self.low_theta, self.high_theta,\
            self.low_phi, self.high_phi)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True iff
        other is a UniformDirectionDistribution with the same bounds.
        """
        if isinstance(other, UniformDirectionDistribution):
            tol_kwargs = {'rtol': 0., 'atol': 1e-9}
            low_theta_equal =\
                np.isclose(self.low_theta, other.low_theta, **tol_kwargs)
            high_theta_equal =\
                np.isclose(self.high_theta, other.high_theta, **tol_kwargs)
            low_phi_equal =\
                np.isclose(self.low_phi, other.low_phi, **tol_kwargs)
            high_phi_equal =\
                np.isclose(self.high_phi, other.high_phi, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([low_theta_equal, high_theta_equal, low_phi_equal,\
                high_phi_equal, metadata_equal])
        else:
            return False

    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with data about this distribution.
        
        group: hdf5 file group to fill with data about this distribution
        """
        group.attrs['class'] = 'UniformDirectionDistribution'
        DirectionDistribution.fill_hdf5_group(self, group)
        group.attrs['low_theta'] = self.low_theta
        group.attrs['high_theta'] = self.high_theta
        group.attrs['low_phi'] = self.low_phi
        group.attrs['high_phi'] = self.high_phi
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a UniformDirectionDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a UniformDirectionDistribution object created from the
                 information in the given group
        """
        try:
            assert group.attrs['class'] == 'UniformDirectionDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "UniformDirectionDistribution.")
        (metadata, psi_center, pointing_center) =\
            DirectionDistribution.load_generic_properties(group)
        low_theta = group.attrs['low_theta']
        high_theta = group.attrs['high_theta']
        low_phi = group.attrs['low_phi']
        high_phi = group.attrs['high_phi']
        return UniformDirectionDistribution(low_theta=low_theta,\
            high_theta=high_theta, low_phi=low_phi, high_phi=high_phi,\
            pointing_center=pointing_center, psi_center=psi_center,\
            metadata=metadata)

