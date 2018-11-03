"""
File: distpy/distribution/GeneralizedParetoDistribution.py
Author: Keith Tauscher
Date: 28 May 2018

Description: File containing a class representing a generalized form of the
             Pareto distribution. Its cdf is given by
             F(x)=1-(1+((x-mu)/sigma))^(1-alpha) where alpha > 1
"""
import numpy as np
import numpy.random as rand
from ..util import int_types, numerical_types
from .Distribution import Distribution

class GeneralizedParetoDistribution(Distribution):
    """
    Class representing a generalized form of the Pareto distribution. Its cdf
    is given by F(x)=1-(1+((x-mu)/sigma))^(1-alpha) where alpha > 1
    """
    def __init__(self, shape, location=0, scale=1, metadata=None):
        """
        Creates a new GeneralizedParetoDistribution object.
        
        shape: alpha in the CDF, strictly greater than 1
        location: mode (and also lower bound) of distribution
        scale: sigma in the CDF, strictly greater than 0
        """
        self.shape = shape
        self.location = location
        self.scale = scale
        self.metadata = metadata
    
    @property
    def shape(self):
        """
        Property storing the shape parameter, alpha, of this distribution. It
        is strictly greater than 1.
        """
        if not hasattr(self, '_shape'):
            raise AttributeError("shape was referenced before it was set.")
        return self._shape
    
    @shape.setter
    def shape(self, value):
        """
        Setter for the shape parameter, alpha, of this distribution.
        
        value: number greater than 1
        """
        if type(value) in numerical_types:
            if value > 1:
                self._shape = float(value)
            else:
                raise ValueError("shape was set to a number less than 1.")
        else:
            raise TypeError("shape parameter was set to a non-number.")
    
    @property
    def location(self):
        """
        Property storing the location parameter, mu, of this distribution. It
        is strictly greater than 1.
        """
        if not hasattr(self, '_location'):
            raise AttributeError("location was referenced before it was set.")
        return self._location
    
    @location.setter
    def location(self, value):
        """
        Setter for the location parameter, mu, of this distribution.
        
        value: number
        """
        if type(value) in numerical_types:
            self._location = float(value)
        else:
            raise TypeError("location parameter was set to a non-number.")
    
    @property
    def scale(self):
        """
        Property storing the scale parameter, sigma, of this distribution. It
        is strictly greater than 0.
        """
        if not hasattr(self, '_scale'):
            raise AttributeError("scale was referenced before it was set.")
        return self._scale
    
    @scale.setter
    def scale(self, value):
        """
        Setter for the scale parameter, sigma, of this distribution.
        
        value: number greater than 0
        """
        if type(value) in numerical_types:
            if value > 0:
                self._scale = float(value)
            else:
                raise ValueError("scale was set to a number less than 0.")
        else:
            raise TypeError("scale parameter was set to a non-number.")
    
    @property
    def numparams(self):
        """
        Only univariate uniform distributions are included here so numparams
        always returns 1.
        """
        return 1
    
    def draw(self, shape=None, random=rand):
        """
        Draws and returns a value from this distribution using numpy.random.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        random: the random number generator to use (default: numpy.random)
        """
        if shape is None:
            shape = ()
        if type(shape) in int_types:
            shape = (shape,)
        uniforms = random.rand(*shape)
        return self.location +\
            (self.scale * (np.power(uniforms, 1 / (1 - self.shape)) - 1))
    
    @property
    def log_value_constant(self):
        """
        Property storing the constant part of the log of the distribution
        """
        if not hasattr(self, '_log_value_constant'):
            self._log_value_constant = np.log((self.shape - 1) / self.scale)
        return self._log_value_constant
    
    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is value.
        
        point: numerical value of the variable
        """
        if point >= self.location:
            return self.log_value_constant - (self.shape *\
                np.log(1 + ((point - self.location) / self.scale)))
        return -np.inf
    
    def to_string(self):
        """
        Finds and returns a string representation of this distribution.
        """
        return "Pareto({0:.2g}, {1:.2g}, {2:.2g})".format(self.shape,\
            self.location, self.scale)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a GeneralizedParetoDistribution with the same shape, location,
        and scale (down to 1e-9 level) and False otherwise.
        """
        if isinstance(other, GeneralizedParetoDistribution):
            shape_close =\
                np.isclose(self.shape, other.shape, atol=1e-9, rtol=0)
            location_close =\
                np.isclose(self.location, other.location, atol=1e-9, rtol=0)
            scale_close =\
                np.isclose(self.scale, other.scale, atol=0, rtol=1e-9)
            metadata_equal = self.metadata_equal(other)
            return\
                all([shape_close, location_close, scale_close, metadata_equal])
        else:
            return False
    
    @property
    def log_prior_constant(self):
        """
        Property storing the constant in the natural logarithm of this
        distribution's value. It is equal to ln[(alpha - 1) / sigma]
        """
        return np.log((self.shape - 1) / self.scale)
    
    def inverse_cdf(self, cdf):
        """
        Inverse of cumulative distribution function.
        
        cdf: value between 0 and 1
        """
        return self.location +\
            (self.scale * (np.power(cdf, 1 / (1 - self.shape)) - 1))
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
        """
        return self.location
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        return None
    
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
        group.attrs['class'] = 'GeneralizedParetoDistribution'
        group.attrs['shape'] = self.shape
        group.attrs['location'] = self.location
        group.attrs['scale'] = self.scale
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a GeneralizedParetoDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: UniformDistribution object created from the information in the
                 given group
        """
        try:
            assert group.attrs['class'] == 'GeneralizedParetoDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "GeneralizedParetoDistribution.")
        metadata = Distribution.load_metadata(group)
        shape = group.attrs['shape']
        location = group.attrs['location']
        scale = group.attrs['scale']
        return GeneralizedParetoDistribution(shape, location=location,\
            scale=scale, metadata=metadata)
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivative of log_value(point) with respect to the
        parameter.
        
        point: single number at which to evaluate the derivative
        
        returns: returns single number representing derivative of log value
        """
        return self.shape / ((self.location - point) - self.scale)
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented. Since it has been implemented, it returns True.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivative of log_value(point) with respect to the
        parameter.
        
        point: single value
        
        returns: single number representing second derivative of log value
        """
        return self.shape / (((self.location - point) - self.scale) ** 2)
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        return GeneralizedParetoDistribution(self.shape, self.location,\
            self.scale)

