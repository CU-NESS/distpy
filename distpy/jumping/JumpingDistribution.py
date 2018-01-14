"""
File: distpy/jumping/JumpingDistribution.py
Author: Keith Tauscher
Date: 20 Dec 2017

Description: File containing base class for all jumping distributions.
"""
from ..util import Savable

def raise_cannot_instantiate_jumping_distribution_error():
    raise NotImplementedError("Some part of JumpingDistribution class was " +\
        "not implemented by subclass or Distribution is being instantiated.")

class JumpingDistribution(Savable):
    """
    This class exists for error catching. Since it exists as
    a superclass of all the jumping distributions, one can call
    isinstance(to_check, JumpingDistribution) to see if to_check is indeed a
    kind of jumping distribution.
    
    All subclasses of this one will implement
    self.draw(source) --- draws a destination point from this distribution
                          given a source point
    self.log_value(source, destination) --- computes the difference between the
                                            log pdf of going from source to
                                            destination and the log probability
                                            of going from destination to source
    self.numparams --- property, not function, storing number of parameters
    self.__eq__(other) --- checks for equality with another object
    self.fill_hdf5_group(group) --- fills given hdf5 group with data from
                                    distribution
    
    In draw() and log_value(), point is a configuration. It could be a
    single number for a univariate distribution or a numpy.ndarray for a
    multivariate distribution.
    """
    def draw(self, source):
        """
        Draws a destination point from this jumping distribution given a source
        point. Must be implemented by any base class.
        
        source: if this JumpingDistribution is univariate, source should be a
                                                           single number
                otherwise, source should be numpy.ndarray of shape (numparams,)
        
        returns: either single value (if distribution is 1D) or array of values
        """
        raise_cannot_instantiate_jumping_distribution_error()
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF: ln(f(source->destination))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        raise_cannot_instantiate_jumping_distribution_error()
    
    def log_value_difference(self, source, destination):
        """
        Computes the log-PDF difference:
        ln(f(source->destination)/f(destination->source))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number difference between one-way log-PDF's
        """
        return log_value(source, destination) - log_value(destination, source)
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. It must be implemented by all subclasses.
        """
        raise_cannot_instantiate_jumping_distribution_error()
    
    def __eq__(self, other):
        """
        Tests for equality between this jumping distribution and other. All
        subclasses must implement this function.
        
        other: JumpingDistribution with which to check for equality
        
        returns: True or False
        """
        raise_cannot_instantiate_jumping_distribution_error()
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this jumping
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this jumping
               distribution
        """
        raise_cannot_instantiate_jumping_distribution_error()
    
    def __call__(self, source, destination):
        """
        Alias for log_value_difference function.
        """
        return self.log_value_difference(source, destination)
    
    def __ne__(self, other):
        """
        This merely enforces that (a!=b) equals (not (a==b)) for all jumping
        distribution objects a and b.
        """
        return (not self.__eq__(other))

