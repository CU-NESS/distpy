"""
File: distpy/DirectionDistribution.py
Author: Keith Tauscher
Date: 12 Aug 2017

Description: File containing subclass of Distributions whose support lies on
             the surface of a sphere.
"""
import numpy as np
from .TypeCategories import numerical_types
from .Distribution import Distribution
try:
    import healpy as hp
except:
    have_healpy = False
else:
    have_healpy = True

class NullRotator(object):
    """
    A class which simulates the use of the Rotator class from healpy if the
    user does not have healpy installed. It only returns back what it was
    given. So, it represents the null rotation.
    """
    def __init__(self):
        """
        All NullRotator objects are identical so no preparation is required.
        """
        pass
    
    def __call__(self, theta, phi, **kwargs):
        """
        Parrots the given arguments back, representing the null rotation
        performed on them.
        """
        return (theta, phi)
    
    def I(self, theta, phi, **kwargs):
        """
        Parrots the given arguments back, representing the null rotation
        performed on them.
        """
        return self(theta, phi, **kwargs)

class DirectionDistribution(Distribution):
    """
    Distribution where the variables are angles of a point on the sphere.
    """
    def __init__(self):
        """
        Since DirectionDistribution is not meant to be instantiated directly,
        this initializer just returns an error.
        """
        raise NotImplementedError("DirectionDistribution is not meant to " +\
                                  "be instantiated directly.")
    
    @property
    def numparams(self):
        """
        number of parameters is always 1 for pointing properties because they
        specify a single point on the celestial sphere.
        """
        return 2
    
    @property
    def pointing_center(self):
        """
        Property storing the (latitude, longitude) pointing center in degrees.
        """
        if not hasattr(self, '_pointing_center'):
            raise AttributeError("pointing_center referenced before it was " +\
                                 "set.")
        return self._pointing_center
    
    @pointing_center.setter
    def pointing_center(self, value):
        """
        Setter for the pointing_center. For users without healpy installed,
        value must be (90,0)
        
        value: (latitude, longitude) tuple in degrees (must be (90,0) if healpy
               is not installed)
        """
        value = np.array(value)
        if value.shape == (2,):
            if abs(value[0]) <= 90:
                if (not have_healpy) and (value != (90, 0)):
                    raise ValueError("pointing_center must be (90, 0) " +\
                                     "because healpy is not installed.")
                self._pointing_center = value
            else:
                raise ValueError("latitude given as pointing_center[0] was " +\
                                 "not between -90 and 90 (degrees).")
        else:
            raise ValueError("pointing_center must be a length-2 1D sequence.")
    
    @property
    def theta_center(self):
        """
        Polar spherical coordinate angle (in degrees) of pointing_center.
        """
        if not hasattr(self, '_theta_center'):
            self._theta_center = 90. - self.pointing_center[0]
        return self._theta_center
    
    @property
    def cos_theta_center(self):
        """
        Property storing the cosine of the polar angle of pointing_center.
        """
        if not hasattr(self, '_cos_theta_center'):
            self._cos_theta_center = np.cos(np.radians(self.theta_center))
        return self._cos_theta_center
    
    @property
    def sin_theta_center(self):
        """
        Property storing the sine of the polar angle of pointing_center.
        """
        if not hasattr(self, '_sin_theta_center'):
            self._sin_theta_center = np.sin(np.radians(self.theta_center))
        return self._sin_theta_center
    
    @property
    def phi_center(self):
        """
        Azimuthal spherical coordinate angle (in degrees) of pointing_center.
        """
        if not hasattr(self, '_phi_center'):
            self._phi_center = self.pointing_center[1]
        return self._phi_center
    
    @property
    def psi_center(self):
        """
        Property storing the angle (in degrees) through which the antenna is
        rotated about its boresight.
        """
        if not hasattr(self, '_psi_center'):
            raise AttributeError("psi_center was referenced before it was " +\
                                 "set.")
        return self._psi_center
    
    @psi_center.setter
    def psi_center(self, value):
        """
        Setter for the angle through which the antenna is rotated about its
        boresight.
        
        value: single number in degrees
        """
        if type(value) in numerical_types:
            if (not have_healpy) and (value != 0):
                raise ValueError("psi_center must be 0 because healpy is " +\
                                 "not installed.")
            self._psi_center = value
        else:
            raise TypeError("psi_center given to DirectionDistribution was " +\
                            "not a single number.")
    
    @property
    def rotator(self):
        """
        Property storing the object which rotates to the pointing_center.
        
        returns: healpy.rotator.Rotator object or functional equivalent
        """
        if not hasattr(self, '_rotator'):
            if have_healpy:
                rot_zprime = hp.rotator.Rotator(rot=(-self.phi_center, 0, 0),\
                    deg=True, eulertype='y')
                rot_yprime = hp.rotator.Rotator(rot=(0, self.theta_center, 0),\
                    deg=True, eulertype='y')
                rot_z = hp.rotator.Rotator(rot=(self.psi_center, 0, 0),\
                    deg=True, eulertype='y')
                self._rotator = rot_zprime * rot_yprime * rot_z
            else:
                self._rotator = NullRotator()
                
        return self._rotator
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this distribution.
        
        group: hdf5 file group to which to write data about this distribution
        """
        group.attrs['psi_center'] = self.psi_center
        group.attrs['pointing_center'] = self.pointing_center

