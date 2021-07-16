"""
Module containing base class for all distributions. All subclasses must
implement:

- `Distribution.draw` method: draws one or more random variates from the
distribution
- `Distribution.log_value` method: evaluates the distribution at a given point
- `Distribution.gradient_computable` property: boolean describing whether the
gradient of the log value of this distribution can be computed. If so,
`Distribution.gradient_of_log_value` method should be implemented.
- `Distribution.hessian_computable` property: boolean describing whether the
hessian of the log value of this distribution can be computed. If so,
`Distribution.hessian_of_log_value` method should be implemented.
- `Distribution.numparams` property: the number of parameters described by the
distribution
- `Distribution.mean` property: the mean of the distribution
- `Distribution.variance` property: the (co)variance of the distribution
- `Distribution.minimum` property: the minimum allowable parameter value(s)
- `Distribution.maximum` property: the maximum allowable parameter value(s)
- `Distribution.is_discrete` property: determines whether the distribution is
discrete or continuous
- `Distribution.inverse_cdf` method: finds the inverse of the cumulative
distribution function (only necessary if this distribution is continuous and
univariate) 
- `Distribution.to_string` method: creates a string summary of the distribution
- `Distribution.copy` method: creates a deep copy of the distribution
- `Distribution.__eq__` method: checks for equality with another object
- `Distribution.fill_hdf5_group` method: fills hdf5 file group with info about
this distribution so it can be loaded later
- `Distribution.load_from_hdf5_group` static method: loads a new `Distribution`
of the current subclass

In addition to the above methods and properties, all `Distribution` objects can
be stored with metadata. If this data can be saved (see
`Distribution.save_metadata`), then it can be saved and loaded in hdf5 files as
well.

**File**: $DISTPY/distpy/distribution/Distribution.py  
**Author**: Keith Tauscher  
**Date**: 30 May 2021
"""
import numpy as np
import matplotlib.pyplot as pl
from ..util import Savable, Loadable, save_dictionary, load_dictionary,\
    numerical_types, bool_types, sequence_types

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

cannot_instantiate_distribution_error = NotImplementedError("Some part of " +\
    "Distribution class was not implemented by subclass or Distribution is " +\
    "being directly instantiated.")

class Distribution(Savable, Loadable):
    """
    Base class for all distributions. All subclasses must implement:
    
    - `Distribution.draw` method: draws one or more random variates from the
    distribution
    - `Distribution.log_value` method: evaluates the distribution at a given
    point
    - `Distribution.gradient_computable` property: boolean describing the
    gradient of the log value of this distribution can be computed. If so,
    `Distribution.gradient_of_log_value` method should be implemented.
    - `Distribution.hessian_computable` property: boolean describing the
    hessian of the log value of this distribution can be computed. If so,
    `Distribution.hessian_of_log_value` method should be implemented.
    - `Distribution.numparams` property: the number of parameters described by
    the distribution
    - `Distribution.mean` property: the mean of the distribution
    - `Distribution.variance` property: the (co)variance of the distribution
    - `Distribution.minimum` property: the minimum allowable parameter value(s)
    - `Distribution.maximum` property: the maximum allowable parameter value(s)
    - `Distribution.is_discrete` property: determines whether the distribution
    is discrete or continuous
    - `Distribution.inverse_cdf` method: finds the inverse of the cumulative
    distribution function (only necessary if this distribution is continuous
    and univariate) 
    - `Distribution.to_string` method: creates a string summary of the
    distribution
    - `Distribution.copy` method: creates a deep copy of the distribution
    - `Distribution.__eq__` method: checks for equality with another object
    - `Distribution.fill_hdf5_group` method: fills hdf5 file group with info
    about this distribution so it can be loaded later
    - `Distribution.load_from_hdf5_group` static method: loads a new
    `Distribution` of the current subclass
    """
    def draw(self, shape=None, random=None):
        """
        Draws a point from the distribution. Must be implemented by all
        subclasses.
        
        Parameters
        ----------
        shape : int or tuple or None
            - if None, returns single random variate:
                - if this distribution is univariate, a scalar is returned
                - if this distribution describes \\(p\\) parameters, then a 1D
                array of length \\(p\\) is returned
            - if int, \\(n\\), returns \\(n\\) random variates:
                - if this distribution is univariate, a 1D array of length
                \\(n\\) is returned
                - if this distribution describes \\(p\\) parameters, then a 2D
                array of shape `(n,p)` is returned
            - if tuple of \\(n\\) ints, returns `numpy.prod(shape)` random
            variates:
                - if this distribution is univariate, an \\(n\\)-D array of
                shape `shape` is returned
                - if this distribution describes \\(p\\) parameters, then an
                \\((n+1)\\)-D array of shape `shape+(p,)` is returned
        random : `numpy.random.RandomState`
            the random number generator to use (by default, `numpy.random` is
            used)
        
        Returns
        -------
        variates : float or `numpy.ndarray`
            either single random variates or array of such variates. See
            documentation of `shape` above for type and shape of return value
        """
        raise cannot_instantiate_distribution_error
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point. Must be implemented by all subclasses.
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float
            natural logarithm of the value of this distribution at `point`. If
            \\(f\\) is this distribution's PDF and \\(x\\) is `point`, then
            `value` is \\(\\ln{\\big(f(x)\\big)}\\)
        """
        raise cannot_instantiate_distribution_error
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True, `Distribution.gradient_of_log_value` method
        can be called safely.
        """
        raise cannot_instantiate_distribution_error
    
    def gradient_of_log_value(self, point):
        """
        Computes the gradient (derivative) of the logarithm of the value of
        this distribution at the given point. Must be implemented by all
        subclasses.
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float or `numpy.ndarray`
            gradient of the natural logarithm of the value of this
            distribution. If \\(f\\) is this distribution's PDF and \\(x\\) is
            `point`, then `value` is
            \\(\\boldsymbol{\\nabla}\\ln{\\big(f(x)\\big)}\\):
            
            - if this distribution is univariate, then a float representing the
            derivative is returned
            - if this distribution describes \\(p\\) parameters, then a 1D
            `numpy.ndarray` of length \\(p\\) is returned
        """
        if not self.gradient_computable:
            raise NotImplementedError("The gradient of the log value of " +\
                "this Distribution object has not been implemented.")
        raise cannot_instantiate_distribution_error
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True, `Distribution.hessian_of_log_value` method
        can be called safely.
        """
        raise cannot_instantiate_distribution_error
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this distribution at the given point. Must be implemented by all
        subclasses.
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float or `numpy.ndarray`
            gradient of the natural logarithm of the value of this
            distribution. If \\(f\\) is this distribution's PDF and \\(x\\) is
            `point`, then `value` is \\(\\boldsymbol{\\nabla}\
            \\boldsymbol{\\nabla}^T\\ln{\\big(f(x)\\big)}\\):
            
            - if this distribution is univariate, then a float representing the
            derivative is returned
            - if this distribution describes \\(p\\) parameters, then a 2D
            `numpy.ndarray` that is \\(p\\times p\\) is returned
        """
        if not self.hessian_computable:
            raise NotImplementedError("The hessian of the log value of " +\
                "this Distribution object has not been implemented.")
        raise cannot_instantiate_distribution_error
    
    def __call__(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point. Alias of the `Distribution.log_value` method.
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float
            natural logarithm of the value of this distribution at `point`. If
            \\(f\\) is this distribution's PDF and \\(x\\) is `point`, then
            `value` is \\(\\ln{\\big(f(x)\\big)}\\)
        """
        return self.log_value(point)
    
    @property
    def numparams(self):
        """
        The integer number of parameters described by this distribution. It
        must be implemented by all subclasses.
        """
        raise cannot_instantiate_distribution_error
    
    @property
    def mean(self):
        """
        The mean of this distribution, if implemented.
        
        - if this distribution is univariate, this is a float
        - if this distribution describes \\(p\\) parameters, this is a 1D
        `numpy.ndarray` of length \\(p\\)
        """
        raise cannot_instantiate_distribution_error
    
    @property
    def variance(self):
        """
        The (co)variance of this distribution, if implemented.
        
        - if this distribution is univariate, this is a float variance
        - if this distribution describes \\(p\\) parameters, this is a 2D
        `numpy.ndarray` \\(p\\times p\\) covariance matrix
        """
        raise cannot_instantiate_distribution_error
    
    @property
    def standard_deviation(self):
        """
        The standard deviation of the distribution (only valid if this
        distribution is univariate)
        """
        if not hasattr(self, '_standard_deviation'):
            if self.numparams == 1:
                self._standard_deviation = np.sqrt(self.variance)
            else:
                raise NotImplementedError("standard_deviation is not " +\
                    "defined for multivariate distributions.")
        return self._standard_deviation
    
    def __len__(self):
        """
        Alias for `Distribution.numparams` created so that `len(distribution)`
        can be used to get the number of parameters of a `Distribution` object
        without explicitly referencing `Distribution.numparams`.
        
        Returns
        -------
        numparams : int
            the number of parameters in a Distribution 
        """
        return self.numparams
    
    def to_string(self):
        """
        Finds a string representation of this distribution. It must be
        implemented by all subclasses.
        
        Returns
        -------
        representation : str
            a string summary of this distribution
        """
        raise cannot_instantiate_distribution_error
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        Parameters
        ----------
        other : object
            object with which to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` represents this distribution
        """
        raise cannot_instantiate_distribution_error
    
    def __ne__(self, other):
        """
        Tests for inequality between this distribution and other.
        
        Parameters
        ----------
        other : object
            object with which to check for inequality
        
        Returns
        -------
        result : bool
            False if and only if `other` represents this distribution
        """
        return (not self.__eq__(other))
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution. It must be
        implemented by all subclasses.
        
        - if this distribution is univariate, this is a float
        - if this distribution describes \\(p\\) parameters, this is a 1D array
        of length \\(p\\)
        """
        raise cannot_instantiate_distribution_error
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution. It must be
        implemented by all subclasses.
        
        - if this distribution is univariate, this is a float
        - if this distribution describes \\(p\\) parameters, this is a 1D array
        of length \\(p\\)
        """
        raise cannot_instantiate_distribution_error
    
    @property
    def bounds(self):
        """
        The bounds of this distribution. It merely combines the
        `Distribution.minimum` and `Distribution.maximum` properties.
        
        - if this distribution is univariate, this is a 2-tuple containing the
        minimum and maximum
        - if this distribution describes \\(p\\) parameters, this is a list of
        2-tuples containing the minima and maxima of the parameters
        """
        if self.numparams == 1:
            return (self.minimum, self.maximum)
        else:
            return list(zip(self.minimum, self.maximum))
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False). It must be implemented by all subclasses.
        """
        raise cannot_instantiate_distribution_error
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with information about this
        distribution so that it can be saved later. All subclasses must
        implement this function. Aside from data necessary to load the
        distribution later, each subclass should save the name of its class in
        `group.attrs['class']`.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill with information about this distribution
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        raise cannot_instantiate_distribution_error
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `Distribution` subclass from the given hdf5 file group. All
        subclasses must implement this method if things are to be saved/loaded
        in hdf5 files.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            a `Distribution` was saved
        
        Returns
        -------
        distribution : `Distribution`
            loaded distribution of the current subclass
        """
        raise cannot_instantiate_distribution_error
    
    def copy(self):
        """
        Copies this distribution. This function ignores metadata.
        
        Returns
        -------
        copied : `Distribution`
            a deep copy of this Distribution.
        """
        raise cannot_instantiate_distribution_error

    @property
    def metadata(self):
        """
        Any piece(s) of data which one may want to keep with this
        `Distribution` object. Keep in mind that this can only be saved to an
        hdf5 file if it is a dictionary whose keys are bools, numbers, strings,
        `numpy.ndarray` objects, or `distpy.util.Savable.Savable` objects.
        """
        if not hasattr(self, '_metadata'):
            raise AttributeError("metadata referenced before it was set.")
        return self._metadata
        
    @metadata.setter
    def metadata(self, value):
        """
        Setter for `Distribution.metadata`.
        
        Parameters
        ----------
        value : obj
            any object
        """
        if type(value) is not type(None):
            is_string = isinstance(value, basestring)
            is_number = (type(value) in numerical_types)
            is_bool = (type(value) in bool_types)
            is_dictionary = isinstance(value, dict)
            is_array = isinstance(value, np.ndarray)
            is_savable = isinstance(value, Savable)
            if not any([is_string, is_number, is_bool, is_dictionary,\
                is_array, is_savable]):
                print("distpy: Even though metadata will be stored in " +\
                    "memory, an error will be thrown if fill_hdf5_group is " +\
                    "called because it is unknown how to save this " +\
                    "metadata to disk (i.e. it is not hdf5-able).")
        self._metadata = value
    
    def metadata_equal(self, other):
        """
        Checks to see if `other` has the same metadata as this `Distribution`.
        
        Parameters
        ----------
        other : `Distribution`
            object whose metadata to compare
        
        Returns
        -------
        result : bool
            True if and only if metadata is the same in both `Distribution`
            objects
        """
        try:
            return np.all(self.metadata == other.metadata)
        except:
            return False
    
    def save_metadata(self, group):
        """
        Saves the metadata from this distribution.
        
        Parameters
        ----------
        group : h5py.Group
            the same group with which fill_hdf5_group is being called on a
            `Distribution` subclass
        """
        if type(self.metadata) is not type(None):
            save_dictionary({'metadata': self.metadata},\
                group.create_group('metadata'))
    
    @staticmethod
    def load_metadata(group):
        """
        Loads the metadata saved with the `Distribution.save_metadata` method,
        if any.
        
        Parameters
        ----------
        group : h5py.Group
            the group with which `fill_hdf5_group` was called on a
            `Distribution` subclass
        
        Returns
        -------
        metadata : object or None
            - if no metadata was stored, `metadata` is None
            - otherwise, `metadata` is the metadata that was saved
        """
        if 'metadata' in group:
            metadata_container = load_dictionary(group['metadata'])
            return metadata_container['metadata']
        else:
            return None
    
    @property
    def can_give_confidence_intervals(self):
        """
        Bool describing whether confidence intervals can be returned for this
        distribution. It is usually True if this is a continuous, univariate
        distribution.
        """
        return ((not self.is_discrete) and (self.numparams == 1))
    
    def inverse_cdf(self, probability):
        """
        Finds the value that is larger than `probability` of the variates drawn
        from this distribution.
        
        Parameters
        ----------
        probability : float
            the probability at which to evaluate the inverse cdf
        
        Returns
        -------
        value : float
            `value` is such that the distribution has probability `probability`
            to yield variates below `value`
        """
        if self.can_give_confidence_intervals:
            raise NotImplementedError("inverse_cdf is not implemented even " +\
                "though can_give_confidence_intervals is True.")
        else:
            raise NotImplementedError("inverse_cdf cannot be evaluated for " +\
                "this distribution because it is not a continuous " +\
                "univariate distribution or its inverse_cdf has not been " +\
                "implemented.")
    
    @property
    def median(self):
        """
        The median of this distribution. This is only implemented when
        `Distribution.can_give_confidence_intervals` is True
        """
        if not hasattr(self, '_median'):
            if self.can_give_confidence_intervals:
                self._median = self.inverse_cdf(0.5)
            else:
                raise NotImplementedError("median cannot be determined for " +\
                    "this distribution.")
        return self._median
    
    def left_confidence_interval(self, probability_level):
        """
        Finds the confidence interval furthest to the left. This is only
        implemented when `Distribution.can_give_confidence_intervals` is True.
        
        Parameters
        ----------
        probability_level : float
            the probability with which a random variable with this distribution
            will exist in returned interval
        
        Returns
        -------
        interval: tuple
            `(low, high)`
        """
        if self.can_give_confidence_intervals:
            return (self.inverse_cdf(0), self.inverse_cdf(probability_level))
        else:
            raise ValueError("Confidence intervals cannot be found for " +\
                "this distribution.")
    
    def central_confidence_interval(self, probability_level):
        """
        Finds the confidence interval which has same probability of lying above
        or below interval. This is only implemented when
        `Distribution.can_give_confidence_intervals` is True.
        
        Parameters
        ----------
        probability_level : float
            the probability with which a random variable with this distribution
            will exist in returned interval
        
        Returns
        -------
        interval: tuple
            `(low, high)`
        """
        if self.numparams == 1:
            return (self.inverse_cdf((1 - probability_level) / 2),\
                self.inverse_cdf((1 + probability_level) / 2))
        else:
            raise ValueError("Confidence intervals cannot be found for " +\
                "this distribution.")
    
    def right_confidence_interval(self, probability_level):
        """
        Finds the confidence interval furthest to the right. This is only
        implemented when `Distribution.can_give_confidence_intervals` is True.
        
        Parameters
        ----------
        probability_level : float
            the probability with which a random variable with this distribution
            will exist in returned interval
        
        Returns
        -------
        interval: tuple
            `(low, high)`
        """
        if self.numparams == 1:
            return\
                (self.inverse_cdf(1 - probability_level), self.inverse_cdf(1))
        else:
            raise ValueError("Confidence intervals cannot be found for " +\
                "this distribution.")
    
    def reset(self):
        """
        Resets the distribution to resample. This method exists so that
        conceptual, truly random distributions can be stored alongside
        `distpy.distribution.DeterministicDistribution.DeterministicDistribution`
        objects, which are really just samples.
        """
        pass
    
    def plot(self, x_values, scale_factor=1, center=False, xlabel='',\
        ylabel='', title='', fontsize=24, ax=None, show=False, **kwargs):
        """
        Plots the probability density function of this distribution evaluated
        at the given x values.
        
        Parameters
        ----------
        x_values : `numpy.ndarray`
            1D `numpy.ndarray` of sorted \\(x\\) values at which to plot this
            distribution (if `center` is True, then the distribution is plotted
            at these numbers of standard deviations from the mean)
        scale_factor : float
            pdf values scaled by this factor are plotted
        center : bool
            determines whether numbers of standard deviations from the mean
            (True) are plotted or values themselves (False) are plotted
        xlabel : str
            label to place on x axis
        ylabel : str
            label to place on y axis
        title : str
            title to place on top of plot
        fontsize : int or str
            size of labels and title
        ax : `matplotlib.Axes`
            `matplotlib.Axes` object on which to plot distribution values. If
            None, a new `matplotlib.Axes` object is created on a new
            `matplotlib.Figure` object
        show : bool
            if True, `matplotlib.pyplot.show` is called before this function
            returns
        kwargs : dict
            keyword arguments to pass to the `matplotlib.pyplot.plot` function
            if this is a continuous distribution or the
            `matplotlib.pyplot.scatter` function if this is a discrete
            distribution
        
        Returns
        -------
        axes : `matplotlib.Axes` or None
            - if `show` is False, `axes` is the `matplotlib.Axes` object on
            which plot was made
            - if `show` is True, `axes` is None
        """
        if self.numparams != 1:
            raise NotImplementedError('plot can only be called with 1D ' +\
                'distributions.')
        if center:
            z_values = self.standard_deviation * np.exp([self.log_value(\
                self.mean + (self.standard_deviation * x_value))\
                for x_value in x_values])
        else:
            z_values =\
                np.exp([self.log_value(x_value) for x_value in x_values])
        xlim = (x_values[0], x_values[-1])
        if type(ax) is type(None):
            fig = pl.figure(figsize=(12,9))
            ax = fig.add_subplot(111)
        if self.is_discrete:
            ax.scatter(x_values, z_values * scale_factor, **kwargs)
        else:
            ax.plot(x_values, z_values * scale_factor, **kwargs)
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
