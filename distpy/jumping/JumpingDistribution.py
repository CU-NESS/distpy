"""
File: distpy/jumping/JumpingDistribution.py
Author: Keith Tauscher
Date: 12 Feb 2018

Description: File containing base class for all jumping distributions.
"""
import numpy as np
import matplotlib.pyplot as pl
from ..util import Savable, Loadable

cannot_instantiate_jumping_distribution_error = NotImplementedError("Some " +\
    "part of JumpingDistribution class was not implemented by subclass or " +\
    "Distribution is being instantiated.")

class JumpingDistribution(Savable, Loadable):
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
    def draw(self, source, random=None):
        """
        Draws a destination point from this jumping distribution given a source
        point. Must be implemented by any base class.
        
        source: if this JumpingDistribution is univariate, source should be a
                                                           single number
                otherwise, source should be numpy.ndarray of shape (numparams,)
        random: the random number generator to use (default: numpy.random)
        
        returns: either single value (if distribution is 1D) or array of values
        """
        raise cannot_instantiate_jumping_distribution_error
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF: ln(f(source->destination))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point
        """
        raise cannot_instantiate_jumping_distribution_error
    
    def log_value_difference(self, source, destination):
        """
        Computes the log-PDF difference:
        ln(f(source->destination)/f(destination->source))
        
        source, destination: either single values (if distribution is 1D) or
                             arrays of values
        
        returns: single number difference between one-way log-PDF's
        """
        return self.log_value(source, destination) -\
            self.log_value(destination, source)
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution. It must be implemented by all subclasses.
        """
        raise cannot_instantiate_jumping_distribution_error
    
    def __len__(self):
        """
        Allows user to access the number of parameters in this distribution by
        using the len function and not explicitly referencing the numparams
        property.
        """
        return self.numparams
    
    def __eq__(self, other):
        """
        Tests for equality between this jumping distribution and other. All
        subclasses must implement this function.
        
        other: JumpingDistribution with which to check for equality
        
        returns: True or False
        """
        raise cannot_instantiate_jumping_distribution_error
    
    @property
    def is_discrete(self):
        """
        Property storing boolean describing whether this JumpingDistribution
        describes discrete (True) or continuous (False) variable(s).
        """
        raise cannot_instantiate_jumping_distribution_error
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this jumping
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this jumping
               distribution
        """
        raise cannot_instantiate_jumping_distribution_error
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a JumpingDistribution from the given hdf5 file group. All
        JumpingDistribution subclasses must implement this method if things are
        to be saved in hdf5 files.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this JumpingDistribution was saved
        
        returns: a JumpingDistribution object created from the information in
                 the given group
        """
        raise cannot_instantiate_jumping_distribution_error
    
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
    
    def plot(self, source, x_values, scale_factor=1, xlabel='', ylabel='',\
        title='', fontsize=24, ax=None, show=False, **kwargs):
        """
        Plots the PDF of this distribution evaluated at the given x values.
        
        source: the source point of the jumping distribution.
        x_values: 1D numpy.ndarray of sorted x values at which to evaluate this
                  distribution
        scale_factor: allows for the pdf values to be scaled by a constant
                      (default 1)
        xlabel: label to place on x axis
        ylabel: label to place on y axis
        title: title to place on top of plot
        fontsize: size of labels and title
        ax: Axes object on which to plot distribution values.
            If None, a new Axes object is created on a new Figure object
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        **kwargs: keyword arguments to pass to the matplotlib.pyplot.plot
                  function if this is a continuous distribution or the
                  matplotlib.pyplot.scatter function if this is a discrete
                  distribution
        
        returns: ax if show is False, None otherwise
        """
        if self.numparams != 1:
            raise NotImplementedError('plot can only be called with 1D ' +\
                'distributions.')
        y_values =\
            np.exp([self.log_value(source, x_value) for x_value in x_values])
        xlim = (x_values[0], x_values[-1])
        if type(ax) is type(None):
            fig = pl.figure(figsize=(12,9))
            ax = fig.add_subplot(111)
        if self.is_discrete:
            ax.scatter(x_values, y_values * scale_factor, **kwargs)
        else:
            ax.plot(x_values, y_values * scale_factor, **kwargs)
        ax.set_xlabel(xlabel, size=fontsize)
        ax.set_ylabel(ylabel, size=fontsize)
        ax.set_title(title, size=fontsize)
        if 'label' in kwargs:
            ax.legend(fontsize=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        ax.set_xlim(xlim)
        if show:
            pl.show()
        else:
            return ax

