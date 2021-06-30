"""
Module containing subclass for representing distributions that exist on the
surface of the sphere.

**File**: $DISTPY/distpy/distribution/DirectionDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
from ..util import numerical_types
from .Distribution import Distribution
try:
    import healpy as hp
except:
    have_healpy = False
else:
    have_healpy = True

class NullRotator(object):
    """
    Class which simulates the use of the `healpy.Rotator` class if the user
    does not have healpy installed. It only returns back what it was given. So,
    it represents the null rotation.
    """
    def __call__(self, theta, phi, **kwargs):
        """
        Parrots the given arguments back, representing the null rotation
        performed on them.
        
        Parameters
        ----------
        theta : float or `numpy.ndarray`
            polar angle(s)
        phi : float or `numpy.ndarray`
            azimuthal angle(s)
        kwargs : dict
            unused dictionary for interface similarity
        
        Returns
        -------
        rotated_theta : float or `numpy.ndarray`
            equal to `theta`
        rotated_phi : float or `numpy.ndarray`
            equal to `phi`
        """
        return (theta, phi)
    
    def I(self, theta, phi, **kwargs):
        """
        Parrots the given arguments back, representing the null rotation
        performed on them.
        
        Parameters
        ----------
        theta : float or `numpy.ndarray`
            polar angle(s)
        phi : float or `numpy.ndarray`
            azimuthal angle(s)
        kwargs : dict
            unused dictionary for interface similarity
        
        Returns
        -------
        derotated_theta : float or `numpy.ndarray`
            equal to `theta`
        derotated_phi : float or `numpy.ndarray`
            equal to `phi`
        """
        return self(theta, phi, **kwargs)

class DirectionDistribution(Distribution):
    """
    Subclass for representing distributions that exist on the surface of the
    sphere.
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
        The number of parameters of this `DirectionDistribution`, which is
        always 2 because `DirectionDistribution` objects describe random values
        on the sphere.
        """
        return 2
    
    @property
    def mean(self):
        """
        The mean of this `DirectionDistribution`.
        """
        if not hasattr(self, '_mean'):
            raise NotImplementedError("mean is not implemented for " +\
                "DirectionDistribution classes.")
        return self._mean
    
    @property
    def variance(self):
        """
        The covariance of this `DirectionDistribution`.
        """
        if not hasattr(self, '_variance'):
            raise NotImplementedError("variance is not implemented for " +\
                "DirectionDistribution classes.")
        return self._variance
    
    @property
    def pointing_center(self):
        """
        The `(latitude, longitude)` pointing center in degrees.
        """
        if not hasattr(self, '_pointing_center'):
            raise AttributeError("pointing_center referenced before it was " +\
                "set.")
        return self._pointing_center
    
    @pointing_center.setter
    def pointing_center(self, value):
        """
        Setter for the `DirectionDistribution.pointing_center`.
        
        Parameters
        ----------
        value : tuple
            `(lat, lon)` tuple in degrees (must be (90,0) if healpy is not
            installed)
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
        Polar spherical coordinate angle (in degrees), \\(\\theta\\), from
        `DirectionDistribution.pointing_center`.
        """
        if not hasattr(self, '_theta_center'):
            self._theta_center = 90. - self.pointing_center[0]
        return self._theta_center
    
    @property
    def cos_theta_center(self):
        """
        The cosine of the polar angle, \\(\\cos{\\theta}\\).
        """
        if not hasattr(self, '_cos_theta_center'):
            self._cos_theta_center = np.cos(np.radians(self.theta_center))
        return self._cos_theta_center
    
    @property
    def sin_theta_center(self):
        """
        The sine of the polar angle, \\(\\sin{\\theta}\\).
        """
        if not hasattr(self, '_sin_theta_center'):
            self._sin_theta_center = np.sin(np.radians(self.theta_center))
        return self._sin_theta_center
    
    @property
    def phi_center(self):
        """
        Azimuthal spherical coordinate angle (in degrees), \\(\\phi\\), from
        `DirectionDistribution.pointing_center`.
        """
        if not hasattr(self, '_phi_center'):
            self._phi_center = self.pointing_center[1]
        return self._phi_center
    
    @property
    def psi_center(self):
        """
        The angle, \\(\\psi\\), (in degrees) through which the distribution is
        rotated about `DirectionDistribution.pointing_center`.
        """
        if not hasattr(self, '_psi_center'):
            raise AttributeError("psi_center was referenced before it was " +\
                "set.")
        return self._psi_center
    
    @psi_center.setter
    def psi_center(self, value):
        """
        Setter for `DirectionDistribution.psi_center`.
        
        Parameters
        ----------
        value : float
            single number in degrees
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
        `healpy.rotator.Rotator` object or equivalent which rotates to the
        `DirectionDistribution.pointing_center`.
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
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `DirectionDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['psi_center'] = self.psi_center
        group.attrs['pointing_center'] = self.pointing_center
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_generic_properties(group):
        """
        Loads the properties that all `DirectionDistribution` instances share.
        
        Parameters
        ----------
        group : h5py.Group
            the hdf5 file group on which fill_hdf5_group was called
        
        Returns
        -------
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            metadata saved alongside saved distribution
        psi_center : float
            \\(\\psi\\) Euler angle in degrees
        pointing_center : tuple
            tuple of form `(lat, lon)` in degrees
        """
        metadata = Distribution.load_metadata(group)
        psi_center = group.attrs['psi_center']
        pointing_center = tuple(group.attrs['pointing_center'])
        return (metadata, psi_center, pointing_center)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. Since `DirectionDistribution` points are on the
        sphere, this is false.
        """
        return False 
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. Since `DirectionDistribution` points are on the
        sphere, this is false.
        """
        return False
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution.
        """
        raise NotImplementedError("minimum makes no sense in the context " +\
            "of directions.")
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution.
        """
        raise NotImplementedError("maximum makes no sense in the context " +\
            "of directions.")

