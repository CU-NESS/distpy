"""
File: distpy/distribution/Distribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing base class for all distributions.
"""
import numpy as np
from ..util import Savable, Loadable, save_dictionary, load_dictionary,\
    numerical_types, bool_types, sequence_types

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

cannot_instantiate_distribution_error = NotImplementedError("Some part of " +\
    "Distribution class was not implemented by subclass or Distribution is " +\
    "being directly instantiated.")

class Distribution(Savable, Loadable):
    """
    This class exists for error catching. Since it exists as
    a superclass of all the distributions, one can call
    isinstance(to_check, Distribution) to see if to_check is indeed a kind of
    distribution.
    
    All subclasses of this one will implement
    self.draw() --- draws a point from this distribution
    self.log_value(point) --- computes the log of the value of this
                              distribution at the given point
    self.numparams --- property, not function, storing number of parameters
    self.to_string() --- string summary of this distribution
    self.__eq__(other) --- checks for equality with another object
    self.fill_hdf5_group(group) --- fills given hdf5 group with data from
                                    distribution
    
    In draw() and log_value(), point is a configuration. It could be a
    single number for a univariate distribution or a numpy.ndarray for a
    multivariate distribution.
    """
    def draw(self, shape=None, random=None):
        """
        Draws a point from the distribution. Must be implemented by any base
        class.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        random: the random number generator to use (default: numpy.random)
        
        returns: either single value (if distribution is 1D) or array of values
        """
        raise cannot_instantiate_distribution_error
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point. It must be implemented by all subclasses.
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        raise cannot_instantiate_distribution_error
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented.
        """
        raise cannot_instantiate_distribution_error
    
    def gradient_of_log_value(self, point):
        """
        Computes the derivative(s) of log_value(point) with respect to the
        parameter(s).
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: if distribution is 1D, returns single number representing
                                        derivative of log value
                 else, returns 1D numpy.ndarray containing the N derivatives of
                       the log value with respect to each individual parameter
        """
        if not self.gradient_computable:
            raise NotImplementedError("The gradient of the log value of " +\
                "this Distribution object has not been implemented.")
        raise cannot_instantiate_distribution_error
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented.
        """
        raise cannot_instantiate_distribution_error
    
    def hessian_of_log_value(self, point):
        """
        Computes the second derivative(s) of log_value(point) with respect to
        the parameter(s).
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: if distribution is 1D, returns single number representing
                                        second derivative of log value
                 else, returns 2D square numpy.ndarray with dimension length
                       equal to the number of parameters representing the N^2
                       different second derivatives of the log value
        """
        if not self.hessian_computable:
            raise NotImplementedError("The hessian of the log value of " +\
                "this Distribution object has not been implemented.")
        raise cannot_instantiate_distribution_error
    
    def __call__(self, point):
        """
        Alias for log_value function.
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        return self.log_value(point)
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. It must be implemented by all subclasses.
        """
        raise cannot_instantiate_distribution_error
    
    def to_string(self):
        """
        Returns a string representation of this distribution. It must be
        implemented by all subclasses.
        """
        raise cannot_instantiate_distribution_error
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: Distribution with which to check for equality
        
        returns: True or False
        """
        raise cannot_instantiate_distribution_error
    
    @property
    def minimum(self):
        """
        Property storing the minimum allowable value(s) in this distribution.
        """
        raise cannot_instantiate_distribution_error
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        """
        raise cannot_instantiate_distribution_error
    
    @property
    def bounds(self):
        """
        Property storing the bounds of this distribution. It merely combines
        the minimum and maximum properties.
        """
        if self.numparams == 1:
            return (self.minimum, self.maximum)
        else:
            return [(mn, mx) in zip(self.minimum, self.maximum)]
    
    @property
    def is_discrete(self):
        """
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        raise cannot_instantiate_distribution_error
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with information about this
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this distribution
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        raise cannot_instantiate_distribution_error
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a Distribution from the given hdf5 file group. All Distribution
        subclasses must implement this method if things are to be saved in hdf5
        files.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a Distribution object created from the information in the
                 given group
        """
        raise cannot_instantiate_distribution_error
    
    def copy(self):
        """
        Returns a deep copy of this Distribution. This function ignores
        metadata.
        """
        raise cannot_instantiate_distribution_error
    
    def save_metadata(self, group):
        """
        Saves the metadata from this distribution.
        
        group: the same group with which fill_hdf5_group is being called on a
               Distribution subclass
        """
        if self.metadata is not None:
            save_dictionary({'metadata': self.metadata},\
                group.create_group('metadata'))
    
    @staticmethod
    def load_metadata(group):
        """
        Loads the metadata saved with the save_metadata() function if any (if
        there is no metadata, this function returns None).
        
        group: the group with which fill_hdf5_group was called on a
               Distribution subclass
        
        returns: None (if no metadata was saved) or the metadata saved when
                 fill_hdf5_group method of a Distribution subclass was called
        """
        if 'metadata' in group:
            metadata_container = load_dictionary(group['metadata'])
            return metadata_container['metadata']
        else:
            return None
    
    def __ne__(self, other):
        """
        This merely enforces that (a!=b) equals (not (a==b)) for all
        distribution objects a and b.
        """
        return (not self.__eq__(other))
    
    @property
    def can_give_confidence_intervals(self):
        """
        Confidence intervals for most distributions can be generated as long as
        this distribution describes only one dimension.
        """
        return ((not self.is_discrete) and (self.numparams == 1))
    
    def left_confidence_interval(self, probability_level):
        """
        Finds confidence interval furthest to the left.
        
        probability_level: the probability with which a random variable with
                           this distribution will exist in returned interval
        
        returns: (low, high) interval
        """
        if self.can_give_confidence_intervals:
            return (self.inverse_cdf(0), self.inverse_cdf(probability_level))
        else:
            raise ValueError("Confidence intervals cannot be found for " +\
                "this distribution.")
    
    def central_confidence_interval(self, probability_level):
        """
        Finds confidence interval which has same probability of lying above or
        below interval.
        
        probability_level: the probability with which a random variable with
                           this distribution will exist in returned interval
        
        returns: (low, high) interval
        """
        if self.numparams == 1:
            return (self.inverse_cdf((1 - probability_level) / 2),\
                self.inverse_cdf((1 + probability_level) / 2))
        else:
            raise ValueError("Confidence intervals cannot be found for " +\
                "this distribution.")
    
    def right_confidence_interval(self, probability_level):
        """
        Finds confidence interval furthest to the right.
        
        probability_level: the probability with which a random variable with
                           this distribution will exist in returned interval
        
        returns: (low, high) interval
        """
        if self.numparams == 1:
            return\
                (self.inverse_cdf(1 - probability_level), self.inverse_cdf(1))
        else:
            raise ValueError("Confidence intervals cannot be found for " +\
                "this distribution.")

    @property
    def metadata(self):
        """
        Property storing any piece of data which one may want to keep with this
        Distribution object. Keep in mind that this can only be saved to an
        hdf5 file if it is a dictionary whose keys are Savable objects (see
        distpy.util.Savable.py; i.e. it must implement a function with the
        following signature: self.fill_hdf5_group(group)) or arrays.
        """
        if not hasattr(self, '_metadata'):
            raise AttributeError("metadata referenced before it was set.")
        return self._metadata
        
    @metadata.setter
    def metadata(self, value):
        """
        Setter for metadata to store with this Disribution. Keep in mind that
        the metadata can only be saved in an hdf5 file group if it is a number,
        bool, string, numpy.ndarray of numbers, a Savable object, or a
        dictionary of such objects.
        
        value: any object, but if saving to hdf5 file is desired check
               description above for acceptable types
        """
        if value is not None:
            is_string = isinstance(value, basestring)
            is_number = (type(value) in numerical_types)
            is_bool = (type(value) in bool_types)
            is_dictionary = isinstance(value, dict)
            is_array = isinstance(value, np.ndarray)
            is_savable = isinstance(value, Savable)
            if not any([is_string, is_number, is_bool, is_dictionary,\
                is_array, is_savable]):
                
                if rank == 0:
                    print("distpy: Even though metadata will be stored in memory, an " +\
                        "error will be thrown if fill_hdf5_group is called " +\
                        "because it is unknown how to save this metadata to " +\
                        "disk (i.e. it is not hdf5-able).")
                        
        self._metadata = value
    
    def metadata_equal(self, other):
        """
        Checks to see if other's metadata is the same as self's.
        
        other: a Distribution object
        
        returns: True if metadata is the same in both Distributions,
                 False otherwise
        """
        try:
            return np.all(self.metadata == other.metadata)
        except:
            return False
    
    def reset(self):
        """
        This function exists so that conceptual distributions can be stored
        alongside DeterministicDistributions, which are really just samples.
        """
        pass

