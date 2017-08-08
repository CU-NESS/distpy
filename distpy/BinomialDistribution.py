"""
File: distpy/BinomialDistribution.py
Author: Keith Tauscher
Date: 6 Aug 2017

Description: File containing class representing a binomial distribution.
"""
import numpy as np
import numpy.random as rand
from scipy.special import gammaln as log_gamma
from .TypeCategories import int_types, numerical_types
from .Distribution import Distribution

class BinomialDistribution(Distribution):
    """
    Distribution with support on the non-negative integers up to a maximum
    numbers. It has only two parameters, probability of sucess and number of
    trials. When probability of success is 1/2, this is the distribution of the
    number of heads flipped in the number of trials given.
    """
    def __init__(self, probability_of_success, number_of_trials):
        """
        Initializes new BinomialDistribution with given scale.
        
        probability_of_success: real number in (0, 1)
        number_of_trials: maximum integer in the support of this distribution
        """
        if type(probability_of_success) in numerical_types:
            if (probability_of_success > 0.) and (probability_of_success < 1.):
                self.probability_of_success = probability_of_success
            else:
                raise ValueError("probability_of_success given to " +\
                                 "BinomialDistribution was not between 0 " +\
                                 "and 1.")
        else:
            raise ValueError("probability_of_success given to " +\
                             "BinomialDistribution was not a number.")
        if type(number_of_trials) in int_types:
            if number_of_trials > 0:
                self.number_of_trials = number_of_trials
            else:
                raise ValueError("number_of_trials given to " +\
                                 "BinomialDistribution was not positive.")
        else:
            raise ValueError("number_of_trials given to " +\
                             "BinomialDistribution was not a number.")
    
    @property
    def numparams(self):
        """
        Binomial distribution pdf is univariate so numparams always returns 1.
        """
        return 1
    
    def draw(self, shape=None):
        """
        Draws and returns a value from this distribution using numpy.random.
        
        shape: if None, returns single random variate
                        (scalar for univariate ; 1D array for multivariate)
               if int, n, returns n random variates
                          (1D array for univariate ; 2D array for multivariate)
               if tuple of n ints, returns that many random variates
                                   n-D array for univariate ;
                                   (n+1)-D array for multivariate
        """
        return rand.binomial(self.number_of_trials,\
            self.probability_of_success, size=shape)
    
    def log_value(self, point):
        """
        Evaluates and returns the log of the value of this distribution when
        the variable is point.
        
        point: numerical value of the variable
        """
        if type(point) in int_types:
            if (point >= 0) and (point <= self.number_of_trials):
                n_minus_k = self.number_of_trials - point
                return log_gamma(self.number_of_trials + 1) -\
                    log_gamma(point + 1) - log_gamma(n_minus_k + 1) +\
                    (point * np.log(self.probability_of_success)) +\
                    (n_minus_k * np.log(1 - self.probability_of_success))
            else:
                return -np.inf
        else:
            raise TypeError("point given to BinomialDistribution was not " +\
                            "an integer.")

    def to_string(self):
        """
        Finds and returns a string version of this BinomialDistribution.
        """
        return "Binomial(%.2g,%.i)" %\
            (self.probability_of_success, self.number_of_trials)
    
    def __eq__(self, other):
        """
        Checks for equality of this distribution with other. Returns True if
        other is a BinomialDistribution with the same probability_of_success
        and number_of_trials.
        """
        if isinstance(other, BinomialDistribution):
            p_close = np.isclose(self.probability_of_success,\
                other.probability_of_success, rtol=0, atol=1e-6)
            n_equal = (self.number_of_trials == other.number_of_trials)
            return p_close and n_equal
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this distribution. The
        only thing to save is the common_ratio.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'BinomialDistribution'
        group.attrs['number_of_trials'] = self.number_of_trials
        group.attrs['probability_of_success'] = self.probability_of_success

