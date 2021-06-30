"""
Module containing class representing a Gaussian distribution on the sphere. Its
PDF is represented by: $$f(\\boldsymbol{\\hat{n}}) \\propto\\exp{\\left\\{\
-\\frac{1}{2}\\left[\\frac{\\arccos{\\left(\\boldsymbol{\\hat{n}}\\cdot\
\\boldsymbol{\\hat{n}}_0\\right)}}{\\alpha}\\right]^2\\right\\}}$$

**File**: $DISTPY/distpy/distribution/ABCDEFDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, numerical_types
from .DirectionDistribution import DirectionDistribution
from .UniformDistribution import UniformDistribution

class GaussianDirectionDistribution(DirectionDistribution):
    """
    Class representing a Gaussian distribution on the sphere. Its PDF is
    represented by: $$f(\\boldsymbol{\\hat{n}}) \\propto\\exp{\\left\\{\
    -\\frac{1}{2}\\left[\\frac{\\arccos{\\left(\\boldsymbol{\\hat{n}}\\cdot\
    \\boldsymbol{\\hat{n}}_0\\right)}}{\\alpha}\\right]^2\\right\\}}$$
    """
    def __init__(self, pointing_center=(90, 0), sigma=1, degrees=True,\
        metadata=None):
        """
        Initializes a new `GaussianDirectionDistribution` with the given
        parameter values.
        
        Parameters
        ----------
        pointing_center : tuple
            2-tuple of the form `(latitude, longitude)`, where both are in
            degrees regardless of `degrees` parameter, giving the central point
            of the distribution (i.e. the peak of the Gaussian)
        sigma : float
            1-\\(\\sigma\\) size of distribution
        degrees : bool
            - if True, `sigma` is given in degrees
            - if False, `sigma` is given in radians
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.psi_center = 0
        self.pointing_center = pointing_center
        if degrees:
            self.sigma = np.radians(sigma)
        else:
            self.sigma = sigma
        self.metadata = metadata
    
    @property
    def psi_distribution(self):
        """
        The distribution of the azimuthal angle about `pointing_center`
        (uniform).
        """
        if not hasattr(self, '_psi_distribution'):
            self._psi_distribution = UniformDistribution(0, 2 * np.pi)
        return self._psi_distribution
    
    @property
    def sigma(self):
        """
        The angular scale of this distribution, \\(\\sigma\\), in radians.
        """
        if not hasattr(self, '_sigma'):
            raise AttributeError("sigma was referenced before it was set.")
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        """
        Setter for `GaussianDirectionDistribution.sigma`.
        
        Parameters
        ----------
        value : float
            single positive number (in radians)
        """
        if type(value) in numerical_types:
            if value > 0:
                self._sigma = value
            else:
                raise ValueError("sigma given to " +\
                    "GaussianDirectionDistribution was not positive.")
        else:
            raise TypeError("sigma given to GaussianDirectionDistribution " +\
                "was not a single number.")
    
    @property
    def const_log_value_contribution(self):
        """
        The constant part of the logarithm of the value of the distribution at
        each given point, given by \\(-\\ln{(2\\pi\\sigma^2)}\\).
        """
        if not hasattr(self, '_const_log_value_contribution'):
            self._const_log_value_contribution =\
                -np.log((self.sigma ** 2) * 2 * np.pi)
        return self._const_log_value_contribution
    
    def to_string(self):
        """
        Finds and returns a string version of this
        `GaussianDirectionDistribution` of the form
        `"GaussianDirection(lat, lon, sigma)"`.
        """
        return 'GaussianDirection(({0:.3g}, {1:.3g}), {2:.3g})'.format(\
            self.pointing_center[0], self.pointing_center[1], self.sigma)
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `GaussianDirectionDistribution` at the given point.
        
        Parameters
        ----------
        point : `numpy.ndarray`
            `point` should be a length-2 `numpy.ndarray`
        
        Returns
        -------
        value : float
            natural logarithm of the value of this distribution at `point`. If
            \\(f\\) is this distribution's PDF and \\(x\\) is `point`, then
            `value` is \\(\\ln{\\big(f(x)\\big)}\\)
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
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `GaussianDirectionDistribution`.
        
        Parameters
        ----------
        shape : int or tuple or None
            - if None, returns single random variate as a 1D array of length
            2 is returned
            - if int, \\(n\\), returns \\(n\\) random variates as a 2D
            array of shape `(n,2)` is returned
            - if tuple of \\(n\\) ints, returns `numpy.prod(shape)` random
            variates as an \\((n+1)\\)-D array of shape `shape+(2,)` is
            returned
        random : `numpy.random.RandomState`
            the random number generator to use (by default, `numpy.random` is
            used)
        
        Returns
        -------
        variates : float or `numpy.ndarray`
            either single random variates or array of such variates. See
            documentation of `shape` above for type and shape of return value
        """
        psi_draw = self.psi_distribution.draw(shape=shape, random=random)
        if type(shape) is type(None):
            gamma_draw =\
                self.sigma * np.sqrt(-2 * np.log(1 - random.rand()))
        else:
            if type(shape) in int_types:
                shape = (shape,)
            gamma_draw =\
                self.sigma * np.sqrt(-2 * np.log(1 - random.rand(*shape)))
        theta_draw, phi_draw = self.rotator(gamma_draw, psi_draw)
        return np.stack([90 - np.degrees(theta_draw), np.degrees(phi_draw)],\
            axis=-1)
    
    def __eq__(self, other):
        """
        Checks for equality of this `GaussianDirectionDistribution` with
        `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `GaussianDirectionDistribution`
            with the same defining
            `GaussianDirectionDistribution.theta_center`,
            `GaussianDirectionDistribution.phi_center`, and
            `GaussianDirectionDistribution.sigma`
        """
        if isinstance(other, GaussianDirectionDistribution):
            tol_kwargs = {'rtol': 0., 'atol': 1e-9}
            theta_center_equal =\
                np.isclose(self.theta_center, other.theta_center, **tol_kwargs)
            phi_center_equal =\
                np.isclose(self.phi_center, other.phi_center, **tol_kwargs)
            sigma_equal = np.isclose(self.sigma, other.sigma, **tol_kwargs)
            metadata_equal = self.metadata_equal(other)
            return all([theta_center_equal, phi_center_equal, sigma_equal,\
                metadata_equal])
        else:
            return False

    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `GaussianDirectionDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'GaussianDirectionDistribution'
        DirectionDistribution.fill_hdf5_group(self, group,\
            save_metadata=save_metadata)
        group.attrs['sigma'] = self.sigma
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `GaussianDirectionDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `GaussianDirectionDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'GaussianDirectionDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "GaussianDirectionDistribution.")
        (metadata, psi_center, pointing_center) =\
            DirectionDistribution.load_generic_properties(group)
        sigma = group.attrs['sigma']
        return GaussianDirectionDistribution(pointing_center=pointing_center,\
            sigma=sigma, degrees=False, metadata=metadata)
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `GaussianDirectionDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return GaussianDirectionDistribution(\
            pointing_center=[element for element in self.pointing_center],\
            sigma=self.sigma, degrees=False)

