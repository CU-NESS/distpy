"""
File: distpy/distribution/UniformConditionDistribution.py
Author: Keith Tauscher
Date: 13 Mar 2019

Description: File containing class representing a distribution which takes on
             the log_value 0 (unnormalized) when a condition  is met and -inf
             when the condition is not met.
"""
import numpy as np
from ..util import int_types, bool_types, Expression
from .Distribution import Distribution

class UniformConditionDistribution(Distribution):
    """
    A class representing a distribution which takes on the log_value 0
    (unnormalized) when a condition  is met and -inf when the condition is not
    met.
    """
    def __init__(self, expression, metadata=None, is_discrete=False):
        """
        Initializes a new UniformConditionDistribution
        
        expression: the condition which defines where the log_value of this
                    distribution is finite. expression.num_arguments is the
                    dimension of this distribution
        metadata: data to store alongside this distribution
        is_discrete: True if the variable underlying this distribution is
                     discrete. False otherwise (default False)
        """
        self.expression = expression
        self.is_discrete = is_discrete
        self.metadata = metadata
    
    @property
    def expression(self):
        """
        Property storing the Expression object which takes parameters as inputs
        and produces the model output.
        """
        if not hasattr(self, '_expression'):
            raise AttributeError("expression referenced before it was set.")
        return self._expression
    
    @expression.setter
    def expression(self, value):
        """
        Setter for the Expression object at the core of this model.
        
        value: must be an Expression object
        """
        if isinstance(value, Expression):
            self._expression = value
        else:
            raise TypeError("expression was not an Expression object.")
    
    def draw(self, shape=None, random=None):
        """
        Draws a point from the distribution. Since this Distribution cannot be
        drawn from, this throws a NotImplementedError.
        """
        raise NotImplementedError("UniformConditionDistribution objects " +\
            "cannot be drawn from because there is zero probability of its " +\
            "variate appearing in any given finite interval.")
    
    def log_value(self, point):
        """
        Computes the logarithm of the value of this distribution at the given
        point.
        
        point: either single value (if distribution is 1D) or array of values
        
        returns: single number, logarithm of value of this distribution at the
                 given point. This distribution is not normalized because that
                 would require knowing the exact region in which the condition
                 at the heart of this distribution is True. If one knew that,
                 they wouldn't be using the UniformConditionDistribution class,
                 they'd use a more specific class.
        """
        if self.numparams == 1:
            return (0. if self.expression(point) else -np.inf)
        else:
            return (0. if self.expression(*point) else -np.inf)
    
    @property
    def gradient_computable(self):
        """
        Property which stores whether the gradient of the given distribution
        has been implemented.
        """
        return True
    
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
        return np.zeros(self.numparams)
    
    @property
    def hessian_computable(self):
        """
        Property which stores whether the hessian of the given distribution
        has been implemented.
        """
        return True
    
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
        return np.zeros((self.numparams, self.numparams))
    
    @property
    def numparams(self):
        """
        Property storing the integer number of parameters described by this
        distribution.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = self.expression.num_arguments
        return self._numparams
    
    def __eq__(self, other):
        """
        Tests for equality between this distribution and other. All subclasses
        must implement this function.
        
        other: Distribution with which to check for equality
        
        returns: True or False
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
        Property storing a boolean describing whether this distribution is
        discrete (True) or continuous (False).
        """
        if not hasattr(self, '_is_discrete'):
            raise AttributeError("is_discrete referenced before it was set.")
        return self._is_discrete
    
    @is_discrete.setter
    def is_discrete(self, value):
        """
        Setter for whether this distribution is discrete or continuous (the
        form itself does not determine this since this distribution cannot be
        drawn from).
        
        value: must be a bool (True for discrete, False for continuous)
        """
        if type(value) in bool_types:
            self._is_discrete = value
        else:
            raise TypeError("is_discrete was set to a non-bool.")
    
    def fill_hdf5_group(self, group, save_metadata=True):
        """
        Fills the given hdf5 file group with information about this
        distribution. All subclasses must implement this function.
        
        group: hdf5 file group to fill with information about this distribution
        save_metadata: if True, attempts to save metadata alongside
                                distribution and throws error if it fails
                       if False, metadata is ignored in saving process
        """
        group.attrs['class'] = 'UniformConditionDistribution'
        group.attrs['is_discrete'] = self.is_discrete
        self.expression.fill_hdf5_group(group.create_group('expression'))
        if save_metadata:
            self.save_metadata(group)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a UniformConditionDistribution from the given hdf5 file group.
        
        group: the same hdf5 file group which fill_hdf5_group was called on
               when this Distribution was saved
        
        returns: a UniformConditionDistribution object created from the
                 information in the given group
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
        Property storing the minimum allowable value(s) in this distribution.
        The one for this Distribution is not known, so it set to None.
        """
        return None
    
    @property
    def maximum(self):
        """
        Property storing the maximum allowable value(s) in this distribution.
        The one for this Distribution is not known, so it set to None.
        """
        return None
    
    @property
    def can_give_confidence_intervals(self):
        """
        Confidence intervals for most distributions can be generated as long as
        this distribution describes only one dimension.
        """
        return False
    
    def copy(self):
        """
        Returns a copy of this Distribution. This function ignores metadata.
        """
        return UniformConditionDistribution(self.expression,\
            is_dicrete=self.is_discrete)

