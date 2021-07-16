"""
Module containing base class for all distributions that describe jumping
through a space. These are used heavily by Markov Chain Monte Carlo (MCMC)
samplers. All subclasses must implement:

- `JumpingDistribution.draw` method: draws one or more random destination
points from the distribution given the source point
- `JumpingDistribution.log_value` method: evaluates the distribution at a given
source and destination
- `JumpingDistribution.log_value_difference` method: finds the difference
between the probability density of
\\(\\boldsymbol{x}\\rightarrow\\boldsymbol{y}\\) and the density of
\\(\\boldsymbol{y}\\rightarrow\\boldsymbol{x}\\). This method has a default
definition which evaluates `JumpingDistribution.log_value` twice, but it can be
overridden to be made more efficient. This is especially useful if the
distribution is symmetric under swaps of source and destination because it
should always return zero.
- `JumpingDistribution.numparams` property: describes the number of parameters
in the space being traversed.
- `JumpingDistribution.__eq__` method: checks for equality of the distribution
with another object
- `JumpingDistribution.is_discrete` property: describes whether the space being
traversed is discrete or continuous.
- `JumpingDistribution.fill_hdf5_group` method: fills a `h5py.Group` with
information about this distribution so it can be loaded later.
- `JumpingDistribution.load_from_hdf5_group` staticmethod: loads a previously
saved distribution.

**File**: $DISTPY/distpy/jumping/JumpingDistribution.py  
**Author**: Keith Tauscher  
**Date**: 12 Feb 2018
"""
import numpy as np
import matplotlib.pyplot as pl
from ..util import Savable, Loadable

cannot_instantiate_jumping_distribution_error = NotImplementedError("Some " +\
    "part of JumpingDistribution class was not implemented by subclass or " +\
    "Distribution is being instantiated.")

class JumpingDistribution(Savable, Loadable):
    """
    Base class for all distributions that describe jumping through a space.
    These are used heavily by Markov Chain Monte Carlo (MCMC) samplers. All
    subclasses must implement:
    
    - `JumpingDistribution.draw` method: draws one or more random destination
    points from the distribution given the source point
    - `JumpingDistribution.log_value` method: evaluates the distribution at a
    given source and destination
    - `JumpingDistribution.log_value_difference` method: finds the difference
    between the probability density of
    \\(\\boldsymbol{x}\\rightarrow\\boldsymbol{y}\\) and the density of
    \\(\\boldsymbol{y}\\rightarrow\\boldsymbol{x}\\). This method has a default
    definition which evaluates `JumpingDistribution.log_value` twice, but it
    can be overridden to be made more efficient. This is especially useful if
    the distribution is symmetric under swaps of source and destination because
    it should always return zero.
    - `JumpingDistribution.numparams` property: describes the number of
    parameters in the space being traversed.
    - `JumpingDistribution.__eq__` method: checks for equality of the
    distribution with another object
    - `JumpingDistribution.is_discrete` property: describes whether the space
    being traversed is discrete or continuous.
    - `JumpingDistribution.fill_hdf5_group` method: fills a `h5py.Group` with
    information about this distribution so it can be loaded later.
    - `JumpingDistribution.load_from_hdf5_group` staticmethod: loads a
    previously saved distribution.
    """
    def draw(self, source, shape=None, random=None):
        """
        Draws a destination point from this jumping distribution given a source
        point. Must be implemented by any base class.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this `JumpingDistribution` is univariate, source should be
            a single number
            - otherwise, source should be `numpy.ndarray` of shape (numparams,)
        shape : None or int or tuple
            - if None, a single destination is returned
                - if this distribution is univariate, a single number is
                returned
                - if this distribution is multivariate, a 1D `numpy.ndarray`
                describing the coordinates of the destination is returned
            - if int \\(n\\), \\(n\\) destinations are returned
                - if this distribution is univariate, a 1D `numpy.ndarray` of
                length \\(n\\) is returned
                - if this distribution describes \\(p\\) dimensions, a 2D
                `numpy.ndarray` is returned whose shape is \\((n,p)\\)
            - if tuple of ints \\((n_1,n_2,\\ldots,n_k)\\),
            \\(\\prod_{m=1}^kn_m\\) destinations are returned
                - if this distribution is univariate, a `numpy.ndarray` of
                shape \\((n_1,n_2,\\ldots,n_k)\\) is returned
                - if this distribution describes \\(p\\) parameters, a
                `numpy.ndarray` of shape \\((n_1,n_2,\\ldots,n_k,p)\\) is
                returned
        random : numpy.random.RandomState
            the random number generator to use (default: `numpy.random`)
        
        Returns
        -------
        drawn : number or numpy.ndarray
            either single value or array of values. See documentation on
            `shape` above for the type of the returned value
        """
        raise cannot_instantiate_jumping_distribution_error
    
    def log_value(self, source, destination):
        """
        Computes the log-PDF of jumping from `source` to `destination`. It must
        be implemented by all subclasses.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this distribution is univariate, `source` must be a number
            - if this distribution describes \\(p\\) parameters, `source` must
            be a 1D `numpy.ndarray` of length \\(p\\)
        destination : number or numpy.ndarray
            - if this distribution is univariate, `destination` must be a
            number
            - if this distribution describes \\(p\\) parameters, `destination`
            must be a 1D `numpy.ndarray` of length \\(p\\)
        
        Returns
        -------
        log_pdf : float
            if the distribution is \\(f(\\boldsymbol{x},\\boldsymbol{y})=\
            \\text{Pr}[\\boldsymbol{y}|\\boldsymbol{x}]\\), `source` is
            \\(\\boldsymbol{x}\\) and `destination` is \\(\\boldsymbol{y}\\),
            then `log_pdf` is given by
            \\(\\ln{f(\\boldsymbol{x},\\boldsymbol{y})}\\)
        """
        raise cannot_instantiate_jumping_distribution_error
    
    def log_value_difference(self, source, destination):
        """
        Computes the difference in the log-PDF of jumping from `source` to
        `destination` and of jumping from `destination` to `source`. While this
        method has a default version, overriding it may provide an efficiency
        benefit.
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this distribution is univariate, `source` must be a number
            - if this distribution describes \\(p\\) parameters, `source` must
            be a 1D `numpy.ndarray` of length \\(p\\)
        destination : number or numpy.ndarray
            - if this distribution is univariate, `destination` must be a
            number
            - if this distribution describes \\(p\\) parameters, `destination`
            must be a 1D `numpy.ndarray` of length \\(p\\)
        
        Returns
        -------
        log_pdf_difference : float
            if the distribution is \\(f(\\boldsymbol{x},\\boldsymbol{y})=\
            \\text{Pr}[\\boldsymbol{y}|\\boldsymbol{x}]\\), `source` is
            \\(\\boldsymbol{x}\\) and `destination` is \\(\\boldsymbol{y}\\),
            then `log_pdf_difference` is given by \\(\\ln{f(\\boldsymbol{x},\
            \\boldsymbol{y})}-\\ln{f(\\boldsymbol{y},\\boldsymbol{x})}\\)
        """
        return self.log_value(source, destination) -\
            self.log_value(destination, source)
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this distribution. It
        must be implemented by all subclasses.
        """
        raise cannot_instantiate_jumping_distribution_error
    
    def __len__(self):
        """
        Allows user to access the number of parameters in this distribution by
        using the `len` function and not explicitly referencing
        `JumpingDistribution.numparams`.
        """
        return self.numparams
    
    def __eq__(self, other):
        """
        Tests for equality between this `JumpingDistribution` and `other`. All
        subclasses must implement this function.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if object is an equivalent `JumpingDistribution`
        """
        raise cannot_instantiate_jumping_distribution_error
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this `JumpingDistribution` describes
        discrete (True) or continuous (False) variable(s).
        """
        raise cannot_instantiate_jumping_distribution_error
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this
        `JumpingDistribution`. All subclasses must implement this function.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill with information about this
            `JumpingDistribution`
        """
        raise cannot_instantiate_jumping_distribution_error
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `JumpingDistribution` from the given hdf5 file group. All
        subclasses must implement this method if things are to be saved in hdf5
        files.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which
            `JumpingDistribution.fill_hdf5_group` was called on when this
            `JumpingDistribution` was saved
        
        Returns
        -------
        loaded : `JumpingDistribution`
            a `JumpingDistribution` object created from the information in the
            given group
        """
        raise cannot_instantiate_jumping_distribution_error
    
    def __call__(self, source, destination):
        """
        Computes the difference in the log-PDF of jumping from `source` to
        `destination` and of jumping from `destination` to `source`. While this
        method has a default version, overriding it may provide an efficiency
        benefit. Alias for `JumpingDistribution.log_value_difference`
        
        Parameters
        ----------
        source : number or numpy.ndarray
            - if this distribution is univariate, `source` must be a number
            - if this distribution describes \\(p\\) parameters, `source` must
            be a 1D `numpy.ndarray` of length \\(p\\)
        destination : number or numpy.ndarray
            - if this distribution is univariate, `destination` must be a
            number
            - if this distribution describes \\(p\\) parameters, `destination`
            must be a 1D `numpy.ndarray` of length \\(p\\)
        
        Returns
        -------
        log_pdf_difference : float
            if the distribution is \\(f(\\boldsymbol{x},\\boldsymbol{y})=\
            \\text{Pr}[\\boldsymbol{y}|\\boldsymbol{x}]\\), `source` is
            \\(\\boldsymbol{x}\\) and `destination` is \\(\\boldsymbol{y}\\),
            then `log_pdf_difference` is given by \\(\\ln{f(\\boldsymbol{x},\
            \\boldsymbol{y})}-\\ln{f(\\boldsymbol{y},\\boldsymbol{x})}\\)
        """
        return self.log_value_difference(source, destination)
    
    def __ne__(self, other):
        """
        Tests for inequality between this `JumpingDistribution` and `other`.
        All subclasses must implement this function.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            False if and only if object is an equivalent `JumpingDistribution`
        """
        return (not self.__eq__(other))
    
    def plot(self, source, x_values, scale_factor=1, xlabel='', ylabel='',\
        title='', fontsize=24, ax=None, show=False, **kwargs):
        """
        Plots the PDF of this distribution evaluated at the given x values.
        This method can only be called on univariate jumping distributions.
        
        Parameters
        ----------
        source : number
            the source point of the jumping distribution (a single number
            because this method assumes the distribution is univariate)
        x_values : numpy.ndarray
            1D `numpy.ndarray` of sorted destination values at which to
            evaluate this distribution
        scale_factor : float
            allows for the pdf values to be scaled by a constant
        xlabel : str
            label to place on x-axis
        ylabel : str
            label to place on y-axis
        title : str
            title to place on top of plot
        fontsize : int or str
            size of labels and title
        ax : matplotlib.Axes
            object on which to plot distribution values. If None, a new
            `matplotlib.Axes` object is created on a new `matplotlib.Figure`
            object
        show : bool
            if True, matplotlib.pyplot.show() is called before this function
            returns
        kwargs : dict
            keyword arguments to pass to the `matplotlib.pyplot.plot` function
            if this is a continuous distribution or the
            `matplotlib.pyplot.scatter` function if this is a discrete
            distribution
        
        Returns
        -------
        ax : None or matplotlib.Axes
            - if show is False, the `matplotlib.Axes` on which the plot was
            made is returned
            - if show is True, `None` is returned
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

