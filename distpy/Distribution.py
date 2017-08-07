"""
File: distpy/Distribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing base class for all distributions.
"""
import h5py

def raise_cannot_instantiate_distribution_error():
    raise NotImplementedError("Some part of Distribution class was not " +\
                              "implemented by subclass or Distribution is " +\
                              "being instantiated.")

class Distribution():
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
    def draw(self):
        """
        Draws a point from the distribution. Must be implemented by any base
        class.
        
        returns: either single value (if distribution is 1D) or array of values
        """
        raise_cannot_instantiate_distribution_error()
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point. It must be implemented by all subclasses.
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        raise_cannot_instantiate_distribution_error()
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. It must be implemented by all subclasses.
        """
        raise_cannot_instantiate_distribution_error()
    
    def to_string(self):
        """
        Returns a string representation of this distribution. It must be
        implemented by all subclasses.
        """
        raise_cannot_instantiate_distribution_error()
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: Distribution with which to check for equality
        
        returns: True or False
        """
        raise_cannot_instantiate_distribution_error()
    
    def fill_hdf5_group(group):
        """
        Fills the given hdf5 file group with information about this
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this distribution
        """
        raise_cannot_instantiate_distribution_error()
    
    def __ne__(self, other):
        """
        This merely enforces that (a!=b) equals (not (a==b)) for all
        distribution objects a and b.
        """
        return (not self.__eq__(other))
    
    def save(self, file_name):
        """
        Saves this distribution in an hdf5 file using the distribution's
        fill_hdf5_group function.
        
        file_name: name of hdf5 file to write
        """
        hdf5_file = h5py.File(file_name, 'w')
        self.fill_hdf5_group(hdf5_file)
        hdf5_file.close()

