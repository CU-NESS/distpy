"""
File: distpy/distribution/LinearDirectionDistribution.py
Author: Keith Tauscher
Date: 3 Oct 2018

Description: File containing a class representing a linear distribution
             existing on the surface of a sphere.
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
    Class representing a linear distribution existing on the surface of a
    sphere.
    """
    def __init__(self, central_pointing, phase_delayed_pointing,\
        angle_distribution, metadata=None):
        """
        Creates a new UniformDistribution with the given range.
        
        central_pointing: 
        phase_delayed_pointing: 
        angle_distribution: Distribution object determining how to draw values
                            of the angle around the great circle (with respect
                            to 0 at the central pointing and pi/2 at the phase
                            delayed pointing
        metadata: any data wished to be stored alongside this distribution
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
        Property storing the central pointing of the distribution.
        """
        if not hasattr(self, '_central_pointing'):
            raise AttributeError("central_pointing was referenced before " +\
                "it was set.")
        return self._central_pointing
    
    @central_pointing.setter
    def central_pointing(self, value):
        """
        Setter for the central pointing. This corresponds to an angle of 0.
        
        value: sequence of 2 numbers, (latitude, longitude) in degrees
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
        Property storing the vector which points from the center of the sphere
        to the central pointing.
        """
        if not hasattr(self, '_central_pointing_vector'):
            (latitude, longitude) = self.central_pointing
            self._central_pointing_vector =\
                hp.pixelfunc.ang2vec(longitude, latitude, lonlat=True)
        return self._central_pointing_vector
    
    @property
    def phase_delayed_pointing(self):
        """
        Property storing the phase-delayed pointing of the distribution.
        """
        if not hasattr(self, '_phase_delayed_pointing'):
            raise AttributeError("phase_delayed_pointing was referenced " +\
                "before it was set.")
        return self._phase_delayed_pointing
    
    @phase_delayed_pointing.setter
    def phase_delayed_pointing(self, value):
        """
        Setter for a pointing between central_pointing and its antipodal point
        in the direction of increasing angle.
        
        value: sequence of 2 numbers, (latitude, longitude) in degrees
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
        Property storing the vector which points from the center of the sphere
        to the phase delayed pointing.
        """
        if not hasattr(self, '_phase_delayed_pointing_vector'):
            (latitude, longitude) = self.phase_delayed_pointing
            self._phase_delayed_pointing_vector =\
                hp.pixelfunc.ang2vec(longitude, latitude, lonlat=True)
        return self._phase_delayed_pointing_vector
    
    @property
    def corrected_phase_delayed_pointing_vector(self):
        """
        A vector which is precisely pi/2 in phase ahead of central_pointing.
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
        Property storing the 1D distribution of angles to draw in radians.
        """
        if not hasattr(self, '_angle_distribution'):
            raise AttributeError("angle_distribution was referenced before " +\
                "it was set.")
        return self._angle_distribution
    
    @angle_distribution.setter
    def angle_distribution(self, value):
        """
        Setter for the distribution from which angles are drawn.
        
        value: Distribution object describing exactly one parameter (angle in
               radians)
        """
        if isinstance(value, Distribution):
            if value.numparams == 1:
                self._angle_distribution = value
            else:
                raise ValueError("angle_distribution was multivariate.")
        else:
            raise TypeError("angle_distribution was set to a " +\
                "non-Distribution.")
    
    @property
    def numparams(self):
        """
        Pointing directions involve 2 numbers, so numparams is 2. However,
        this distribution is degenerate and the 2 numbers are mutually
        dependent.
        """
        return 2
    
    def draw(self, shape=None, random=rand):
        """
        Draws and returns a value from this distribution using numpy.random.
        
        shape: if None, returns single random variate in a length-2 array
               if int, n, returns n random variates in an array of shape (n,2)
               if tuple of n ints, returns that many random variates in an
                                   array of shape (shape+(2,))
        random: the random number generator to use (default: numpy.random)
        """
        none_shape = (shape is None)
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
        Evaluates and returns the log of the value of this distribution when
        the variable is value.
        
        point: numerical value of the variable
        """
        raise NotImplementedError("log_value is not implemented because " +\
            "the distribution is degenerate: the two angles which are " +\
            "returned are not independent of each other.")
    
    def to_string(self):
        """
        Finds and returns a string representation of this distribution.
        """
        return "LinDir"
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a LinearDirectionDistribution with the same orientation and
        angle distribution and False otherwise.
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
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        return False
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data from this distribution. All
        that needs to be saved is the class name and high and low values.
        
        group: hdf5 file group to fill
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                      if False, metadata is ignored in saving process
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
        Loads a LinearDirectionDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        angle_distribution_class: the Distribution subclass which created the
                                  angle_distribution of the
                                  LinearDirectionDistribution to be loaded.
        
        returns: LinearDirectionDistribution object created from the
                 information in the given group
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
        Property which stores whether the gradient of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return False
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivative of log_value(point) with respect to the
        parameter.
        
        point: single number at which to evaluate the derivative
        
        returns: returns single number representing derivative of log value
        """
        raise NotImplementedError("gradient_of_log_value can't be defined " +\
            "because this distribution has 1 degree of freedom but exists " +\
            "in 2 dimensions.")
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return False
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivative of log_value(point) with respect to the
        parameter.
        
        point: single value
        
        returns: single number representing second derivative of log value
        """
        raise NotImplementedError("hessian_of_log_value can't be defined " +\
            "because this distribution has 1 degree of freedom but exists " +\
            "in 2 dimensions.")
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return LinearDirectionDistribution(\
            np.array(self.central_pointing).copy(),\
            np.array(self.phase_delayed_pointing).copy(),\
            self.angle_distribution.copy())

