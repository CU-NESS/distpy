"""
Module containing class representing a distribution. Its PDF is
represented by: $$f(x)\\propto\\begin{cases} 1 & p(x)\\text{ true} \\\\\
0 & p(x)\\text{ false} \\end{cases},$$ where \\(p(x)\\) is a proposition
depending on \\(x\\).

**File**: $DISTPY/distpy/distribution/UniformConditionDistribution.py  
**Author**: Keith Tauscher  
**Date**: 31 May 2021
"""
import numpy as np
from ..util import int_types, bool_types, Expression
from .Distribution import Distribution

class UniformConditionDistribution(Distribution):
    """
    Class representing a distribution. Its PDF is represented by:
    $$f(x)\\propto\\begin{cases} 1 & p(x)\\text{ true} \\\\ 0 &\
    p(x)\\text{ false} \\end{cases},$$ where \\(p(x)\\) is a proposition
    depending on \\(x\\).
    """
    def __init__(self, expression, is_discrete=False, metadata=None):
        """
        Initializes a new `BinomialDistribution` with the given parameter
        values.
        
        Parameters
        ----------
        expression : `distpy.util.Expression.Expression`
            expression, \\(p(x)\\) determining where this distribution is
            nonzero
        is_discrete : bool
            bool determining whether this distribution should be considered
            discrete
        metadata : number or str or dict or `distpy.util.Savable.Savable`
            data to store alongside this distribution.
        """
        self.expression = expression
        self.is_discrete = is_discrete
        self.metadata = metadata
    
    @property
    def expression(self):
        """
        The `distpy.util.Expression.Expression` object which takes parameters
        as inputs and produces the model output.
        """
        if not hasattr(self, '_expression'):
            raise AttributeError("expression referenced before it was set.")
        return self._expression
    
    @expression.setter
    def expression(self, value):
        """
        Setter for `UniformConditionDistribution.expression`.
        
        Parameters
        ----------
        value : `distpy.util.Expression.Expression`
            condition
        """
        if isinstance(value, Expression):
            self._expression = value
        else:
            raise TypeError("expression was not an Expression object.")
    
    def draw(self, shape=None, random=None):
        """
        Draws a point from this `UniformConditionDistribution`. Since it
        cannot be drawn from, this throws a NotImplementedError.
        """
        raise NotImplementedError("UniformConditionDistribution objects " +\
            "cannot be drawn from because there is zero probability of its " +\
            "variate appearing in any given finite interval.")
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this
        `UniformConditionDistribution` at the given point.
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float
            0 if the `UniformConditionDistribution.expression` is True at
            `point`, -np.inf otherwise
        """
        if self.numparams == 1:
            return (0. if self.expression(point) else -np.inf)
        else:
            return (0. if self.expression(*point) else -np.inf)
    
    @property
    def gradient_computable(self):
        """
        Boolean describing whether the gradient of the given distribution has
        been implemented. If True,
        `UniformConditionDistribution.gradient_of_log_value` method can be
        called safely.
        """
        return True
    
    def gradient_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `UniformConditionDistribution` at the given point.
        
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
        if self.numparams == 1:
            return 0.
        else:
            return np.zeros(self.numparams)
    
    @property
    def hessian_computable(self):
        """
        Boolean describing whether the hessian of the given distribution has
        been implemented. If True,
        `UniformConditionDistribution.hessian_of_log_value` method can be
        called safely.
        """
        return True
    
    def hessian_of_log_value(self, point):
        """
        Computes the hessian (second derivative) of the logarithm of the value
        of this `UniformConditionDistribution` at the given point.
        
        Parameters
        ----------
        point : float or `numpy.ndarray`
            - if this distribution is univariate, `point` should be a scalar
            - if this distribution describes \\(p\\) parameters, `point` should
            be a length-\\(p\\) `numpy.ndarray`
        
        Returns
        -------
        value : float or `numpy.ndarray`
            hessian of the natural logarithm of the value of this
            distribution. If \\(f\\) is this distribution's PDF and \\(x\\) is
            `point`, then `value` is \\(\\boldsymbol{\\nabla}\
            \\boldsymbol{\\nabla}^T\\ln{\\big(f(x)\\big)}\\):
            
            - if this distribution is univariate, then a float representing the
            derivative is returned
            - if this distribution describes \\(p\\) parameters, then a 2D
            `numpy.ndarray` that is \\(p\\times p\\) is returned
        """
        if self.numparams == 1:
            return 0.
        else:
            return np.zeros((self.numparams, self.numparams))
    
    @property
    def numparams(self):
        """
        The number of parameters of this `UniformConditionDistribution`.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = self.expression.num_arguments
        return self._numparams
    
    @property
    def mean(self):
        """
        The mean of the `UniformConditionDistribution` class is not
        implemented.
        """
        if not hasattr(self, '_mean'):
            raise NotImplementedError("mean is not implemented for " +\
                "UniformConditionDistribution class.")
        return self._mean
    
    @property
    def variance(self):
        """
        The variance of the `UniformConditionDistribution` class is not
        implemented.
        """
        if not hasattr(self, '_variance'):
            raise NotImplementedError("variance is not implemented for " +\
                "UniformConditionDistribution class.")
        return self._variance
    
    def __eq__(self, other):
        """
        Checks for equality of this `UniformConditionDistribution` with
        `other`.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `UniformConditionDistribution`
            with the same `UniformConditionDistribution.expression` and
            `UniformConditionDistribution.is_discrete`
        """
        if not isinstance(other, UniformConditionDistribution):
            return False
        if self.expression != other.expression:
            return False
        if self.is_discrete != other.is_discrete:
            return False
        return self.metadata_equal(other)
    
    @property
    def is_discrete(self):
        """
        Boolean describing whether this distribution is discrete (True) or
        continuous (False).
        """
        if not hasattr(self, '_is_discrete'):
            raise AttributeError("is_discrete referenced before it was set.")
        return self._is_discrete
    
    @is_discrete.setter
    def is_discrete(self, value):
        """
        Setter for `UniformConditionDistribution.is_discrete`
        
        Parameters
        ----------
        value : bool
            True or False
        """
        if type(value) in bool_types:
            self._is_discrete = value
        else:
            raise TypeError("is_discrete was set to a non-bool.")

    def to_string(self):
        """
        Finds and returns a string version of this
        `UniformConditionDistribution` of the form `"UniformCondition"`.
        """
        return "UniformCondition"
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with data about this
        `UniformConditionDistribution` so that it can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill
        save_metadata : bool
            - if True, attempts to save metadata alongside distribution and
            throws error if it fails
            - if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'UniformConditionDistribution'
        group.attrs['is_discrete'] = self.is_discrete
        self.expression.fill_hdf5_group(group.create_group('expression'))
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `UniformConditionDistribution` from the given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            the same hdf5 file group which fill_hdf5_group was called on when
            this Distribution was saved
        
        Returns
        -------
        distribution : `UniformConditionDistribution`
            distribution created from the information in the given group
        """
        try:
            assert group.attrs['class'] == 'UniformConditionDistribution'
        except:
            raise TypeError("The given hdf5 file doesn't seem to contain a " +\
                "UniformConditionDistribution.")
        metadata = Distribution.load_metadata(group)
        expression = Expression.load_from_hdf5_group(group['expression'])
        is_discrete = group.attrs['is_discrete']
        return UniformConditionDistribution(expression,\
            is_discrete=is_discrete, metadata=metadata)
    
    @property
    def minimum(self):
        """
        The minimum allowable value(s) in this distribution. The one for this
        `UniformConditionDistribution` is not known, so it set to None.
        """
        return (None if (self.numparams == 1) else ([None] * self.numparams))
    
    @property
    def maximum(self):
        """
        The maximum allowable value(s) in this distribution. The one for this
        `UniformConditionDistribution` is not known, so it set to None.
        """
        return (None if (self.numparams == 1) else ([None] * self.numparams))
    
    @property
    def can_give_confidence_intervals(self):
        """
        Unnormalized distributions do not support confidence intervals.
        """
        return False
    
    def copy(self):
        """
        Copies this distribution.
        
        Returns
        -------
        copied : `UniformConditionDistribution`
            a deep copy of this distribution, ignoring metadata.
        """
        return UniformConditionDistribution(self.expression,\
            is_dicrete=self.is_discrete)

