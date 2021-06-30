"""
Module containing class representing a distribution that exists along a
great circle on the sphere. Its PDF is represented by:
$$f(\\boldsymbol{\\hat{n}})=\\delta(\\boldsymbol{\\hat{n}}\\cdot\
\\boldsymbol{\\hat{n}}_{\\text{pole}})\\ g\\left\\{\\text{arg}\\left[\
\\boldsymbol{\\hat{n}}\\cdot\\left(\\boldsymbol{\\hat{n}}_{\\text{center}} +\
i\\boldsymbol{\\hat{n}}_{\\text{pole}}\\times\
\\boldsymbol{\\hat{n}}_{\\text{center}}\\right)\\right]\\right\\},$$ where
\\(g\\) is the PDF of an angle distribution and \\(\\delta(x)\\) is the Dirac
delta function.

**File**: $DISTPY/distpy/distribution/LinearDirectionDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, sequence_types, numerical_types
from .Distribution import Distribution
try:
    import healpy as hp
except:
    have_healpy = False
else:
    have_healpy = True

class LinearDirectionDistribution(Distribution):
    """
    Class representing a distribution that exists along a great circle on the
    sphere. Its PDF is represented by: $$f(\\boldsymbol{\\hat{n}})=\
    \\delta(\\boldsymbol{\\hat{n}}\\cdot\
    \\boldsymbol{\\hat{n}}_{\\text{pole}})\\ g\\left\\{\\text{arg}\\left[\
    \\boldsymbol{\\hat{n}}\\cdot\\left(\\boldsymbol{\\hat{n}}_{\\text{center}}\
    +i\\boldsymbol{\\hat{n}}_{\\text{pole}}\\times\
    \\boldsymbol{\\hat{n}}_{\\text{center}}\\right)\\right]\\right\\},$$ where
    \\(g\\) is the PDF of an angle distribution and \\(\\delta(x)\\) is the
    Dirac delta function.
    """
    def __init__(self, central_pointing, phase_delayed_pointing,\
        angle_distribution, metadata=None):
        """
        Initializes a new `LinearDirectionDistribution` with the given
        parameter values.
        
        Parameters
        ----------
        central_pointing : tuple
            2-tuple of the form `(latitude, longitude)`, where both are given
            in degrees, that is the point corresponding to zero angle.
            `central_pointing` corresponds to
            \\(\\boldsymbol{\\hat{n}}_{\\text{center}}\\)
        phase_delayed_pointing : tuple
            2-tuple of the form `(latitude, longitude)`, where both are given
            in degrees, that is somewhere along the great circle ahead of the
            starting point, `central_pointing`. If the phase delay is
            \\(\\pi/2\\), then `phase_delayed_pointing` corresponds to
            \\(\\boldsymbol{\\hat{n}}_{\\text{pole}}\\times\
            \\boldsymbol{\\hat{n}}_{\\text{center}}\\)
        angle_distribution : `distpy.distribution.Distribution.Distribution`
            the distribution, with PDF \\(g\\), of angles to draw along the
            great circle
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        if not have_healpy:
            raise RuntimeError("LinearDirectionDistribution cannot be " +\
                "initialized unless healpy is installed.")
        self.central_pointing = central_pointing
        self.phase_delayed_pointing = phase_delayed_pointing
        self.angle_distribution = angle_distribution
        self.metadata = metadata
    
    @property
    def central_pointing(self):
        """
        The central pointing of the distribution in a 2-tuple of form
        `(lat, lon)`.
        """
        if not hasattr(self, '_central_pointing'):
            raise AttributeError("central_pointing was referenced before " +\
                "it was set.")
        return self._central_pointing
    
    @central_pointing.setter
    def central_pointing(self, value):
        """
        Setter for `LinearDirectionDistribution.central_pointing`.
        
        Parameters
        ----------
        value : sequence
            sequence of 2 numbers, `(lat, lon)` in degrees
        """
        if type(value) in sequence_types:
            if len(value) == 2:
                if all([(type(elem) in numerical_types) for elem in value]):
                    self._central_pointing = np.array(value)
                else:
                    raise TypeError("At least one of the elements of " +\
                        "central_pointing was not a number.")
            else:
                raise ValueError("central_pointing was set to a sequence " +\
                    "that does not have length 2.")
        else:
            raise TypeError("central_pointing was set to a non-sequence.")
    
    @property
    def central_pointing_vector(self):
        """
        The vector which points from the center of the sphere to
        `LinearDirectionDistribution.central_pointing`.
        """
        if not hasattr(self, '_central_pointing_vector'):
            (latitude, longitude) = self.central_pointing
            self._central_pointing_vector =\
                hp.pixelfunc.ang2vec(longitude, latitude, lonlat=True)
        return self._central_pointing_vector
    
    @property
    def phase_delayed_pointing(self):
        """
        The phase-delayed pointing of the distribution, i.e. a pointing that is
        along the great circle from
        `LinearDirectionDistribution.central_pointing`.
        """
        if not hasattr(self, '_phase_delayed_pointing'):
            raise AttributeError("phase_delayed_pointing was referenced " +\
                "before it was set.")
        return self._phase_delayed_pointing
    
    @phase_delayed_pointing.setter
    def phase_delayed_pointing(self, value):
        """
        Setter for `LinearDirectionDistribution.phase_delayed_pointing`.
        
        Parameters
        ----------
        value : tuple
            sequence of 2 numbers, `(lat, lon)` in degrees
        """
        if type(value) in sequence_types:
            if len(value) == 2:
                if all([(type(elem) in numerical_types) for elem in value]):
                    self._phase_delayed_pointing = np.array(value)
                else:
                    raise TypeError("At least one of the elements of " +\
                        "phase_delayed_pointing was not a number.")
            else:
                raise ValueError("phase_delayed_pointing was set to a " +\
                    "sequence that does not have length 2.")
        else:
            raise TypeError("phase_delayed_pointing was set to a " +\
                "non-sequence.")
    
    @property
    def phase_delayed_pointing_vector(self):
        """
        The vector which points from the center of the sphere to
        `LinearDirectionDistribution.phase_delayed_pointing`.
        """
        if not hasattr(self, '_phase_delayed_pointing_vector'):
            (latitude, longitude) = self.phase_delayed_pointing
            self._phase_delayed_pointing_vector =\
                hp.pixelfunc.ang2vec(longitude, latitude, lonlat=True)
        return self._phase_delayed_pointing_vector
    
    @property
    def corrected_phase_delayed_pointing_vector(self):
        """
        A vector which is precisely pi/2 in phase ahead of
        `LinearDirectionDistribution.central_pointing`.
        """
        if not hasattr(self, '_corrected_phase_delayed_pointing_vector'):
            central_lonlat = np.array(self.central_pointing)[-1::-1]
            phase_delayed_lonlat =\
                np.array(self.phase_delayed_pointing)[-1::-1]
            displacement = hp.rotator.angdist(central_lonlat,\
                phase_delayed_lonlat, lonlat=True)
            self._corrected_phase_delayed_pointing_vector =\
                (self.phase_delayed_pointing_vector -\
                (np.cos(displacement) * self.central_pointing_vector)) /\
                np.sin(displacement)
        return self._corrected_phase_delayed_pointing_vector
    
    @property
    def angle_distribution(self):
        """
        The 1D distribution of angles to draw in radians.
        """
        if not hasattr(self, '_angle_distribution'):
            raise AttributeError("angle_distribution was referenced before " +\
                "it was set.")
        return self._angle_distribution
    
    @angle_distribution.setter
    def angle_distribution(self, value):
        """
        Setter for `LinearDirectionDistribution.angle_distribution`.
        
        Parameters
        ----------
        value : `distpy.distribution.Distribution.Distribution`
            distribution describing exactly one parameter (angle in radians)
        """
        if isinstance(value, Distribution):
            if value.numparams == 1:
                self._angle_distribution = value
            else:
                raise ValueError("angle_distribution was multivariate.")
        else:
            raise TypeError("angle_distribution was set to a " +\
                "non-Distribution.")
    
    def draw(self, shape=None, random=rand):
        """
        Draws point(s) from this `LinearDirectionDistribution`.
        
        Parameters
        ----------
        shape : int or tuple or None
            - if None, returns single random variate as a 1D array of length 2
            is returned
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
        none_shape = (type(shape) is type(None))
        if none_shape:
            shape = 1
        if type(shape) in int_types:
            shape = (shape,)
        shape_1D = (len(shape) == 1)
        prodshape = (np.prod(shape),)
        angles = self.angle_distribution.draw(shape=prodshape, random=random)
        cos_angles = np.cos(angles)[:,np.newaxis]
        sin_angles = np.sin(angles)[:,np.newaxis]
        vectors = (self.central_pointing_vector[np.newaxis,:] * cos_angles) +\
            (self.corrected_phase_delayed_pointing_vector[np.newaxis,:] *\
            sin_angles)
        (longitudes, latitudes) = hp.pixelfunc.vec2ang(vectors, lonlat=True)
        pointings = np.stack([latitudes, longitudes], axis=-1)
        if not shape_1D:
            pointings = np.reshape(pointings, shape + (2,))
        if none_shape:
            return pointings[0]
        else:
            return pointings
    
    def log_value(self, point):
        """
        The `LinearDirectionDistribution` is improper, so its log value cannot
        be evaluated.
        """
        raise NotImplementedError("log_value is not implemented because " +\
            "the distribution is degenerate: the two angles which are " +\
            "returned are not independent of each other.")
    
    def to_string(self):
        """
        Finds and returns a string version of this
        `LinearDirectionDistribution` of the form `"LinDir"`.
        """
        return "LinDir"
    
    def __eq__(self, other):
        """
        Checks for equality of this `LinearDirectionDistribution` with `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `LinearDirectionDistribution` with
            the same `LinearDirectionDistribution.central_pointing`,
            `LinearDirectionDistribution.phase_delayed_pointing`, and
            `LinearDirectionDistribution.angle_distribution`
        """
        if isinstance(other, LinearDirectionDistribution):
            tol_kwargs = {'rtol': 0., 'atol': 1e-9}
            central_pointing_close = np.allclose(self.central_pointing,\
                other.central_pointing, **tol_kwargs)
            phase_delayed_pointing_close =\
                np.allclose(self.phase_delayed_pointing,\
                other.phase_delayed_pointing, **tol_kwargs)
            angle_distribution_equal =\
                (self.angle_distribution == other.angle_distribution)
            metadata_equal = self.metadata_equal(other)
            return all([central_pointing_close, phase_delayed_pointing_close,\
                angle_distribution_equal, metadata_equal])
        else:
            return False
    
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
        `LinearDirectionDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'LinearDirectionDistribution'
        group.attrs['central_pointing'] = np.array(self.central_pointing)
        group.attrs['phase_delayed_pointing'] =\
            np.array(self.phase_delayed_pointing)
        self.angle_distribution.fill_hdf5_group(\
            group.create_group('angle_distribution'))
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group, angle_distribution_class):
        """
        Loads a `LinearDirectionDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        angle_distribution_class : class
            class of angle distribution
        
        Returns
        -------
        distribution : `LinearDirectionDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'LinearDirectionDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "LinearDirectionDistribution.")
        metadata = Distribution.load_metadata(group)
        central_pointing = group.attrs['central_pointing']
        phase_delayed_pointing = group.attrs['phase_delayed_pointing']
        angle_distribution = angle_distribution_class.load_from_hdf5_group(\
            group['angle_distribution'])
        return LinearDirectionDistribution(central_pointing,\
            phase_delayed_pointing, angle_distribution, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `LinearDirectionDistribution.gradient_of_log_value` method can be
        called safely.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `LinearDirectionDistribution.hessian_of_log_value` method can be
        called safely.
        """
        return False
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `LinearDirectionDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return LinearDirectionDistribution(\
            np.array(self.central_pointing).copy(),\
            np.array(self.phase_delayed_pointing).copy(),\
            self.angle_distribution.copy())

