"""
File: distpy/jumping/MetropolisHastingsSampler.py
Author: Keith Tauscher (with inspiration from DFM's emcee)
Date: 12 Feb 2018

Description: File containing class implementing an MCMC Sampler class using
             distpy's JumpingDistributionSet objects for the storing of
             proposal distributions.
"""
from __future__ import division
import numpy as np
from ..util import int_types, sequence_types
from .JumpingDistributionSet import JumpingDistributionSet

try:
    from multiprocess import Pool
except ImportError:
    have_multiprocess = False
else:
    have_multiprocess = True

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str
    
def array_from_dictionary(dictionary, parameters):
    """
    Creates an array out of the dictionary.
        
    dictionary: dict with parameters as keys and arrays or numbers as values 
    
    returns: array of ndim 1 greater than values ndim
    """
    return np.array([dictionary[parameter] for parameter in parameters])

def dictionary_from_array(array, parameters):
    """
    Creates a dictionary out of the arrays.
   
    array: numpy.ndarray whose first dimension has length num_parameters
    
    returns: dictionary with parameters as keys
    """
    return {parameter: array[iparameter]\
        for (iparameter, parameter) in enumerate(parameters)}

def choose_destination(input_tuple):
    """
    Chooses a destination point given the source point and its loglikelihood.
    
    input_tuple: (source, lnprob, jumping_distribution_set, logprobability,
                 args, kwargs, parameters, random)
    
    returns: output_tuple: (destination, newlnprob, accepted, random)
    """
    (source, lnprob, jumping_distribution_set, logprobability, args, kwargs,\
        parameters, random) = input_tuple
    source_dict = dictionary_from_array(source, parameters)
    destination_dict =\
        jumping_distribution_set.draw(source_dict, random=random)
    destination = array_from_dictionary(destination_dict, parameters)
    newlnprob = logprobability(destination, *args, **kwargs)
    log_value_difference = jumping_distribution_set.log_value_difference(\
        source_dict, destination_dict)
    diff = newlnprob - lnprob - log_value_difference
    # M-H acceptance ratio
    if diff < 0:
        diff = np.exp(diff) - random.rand()
    if diff > 0:
        return (destination, newlnprob, True, random)
    else:
        return (source, lnprob, False, random)

class DummyPool(object):
    """
    Class representing a dummy-Pool which uses the built-in map function to do
    its mapping.
    """
    def map(self, function, iterable):
        """
        Calls the given function on each element of the given iterable.
        
        function: function to apply to each element of iterable
        iterable: sequence of elements to which to apply function
        
        returns: list where every element is given by the application of the
                 function to the corresponding element of the iterable
        """
        return map(function, iterable)
    
    def close(self):
        """
        Closes the pool (by doing nothing).
        """
        pass

class MetropolisHastingsSampler(object):
    """
    Class implementing an MCMC Sampler class using distpy's
    JumpingDistributionSet objects for the storing of proposal distributions.
    """
    def __init__(self, parameters, num_walkers, logprobability,\
        jumping_distribution_set, num_threads=1, args=[], kwargs={}):
        """
        Initializes a new sampler of the given log probability function.
        
        parameters: names of parameters explored by this sampler
        num_walkers: the number of independent MCMC iterates should be run at once
        logprobability: callable taking parameter array as input
        jumping_distribution_set: JumpingDistributionSet object storing
                                  proposal distributions used to sample given
                                  log probability function
        num_threads: the number of threads to use in log likelihood calculations
                  for walkers. Default: 1, 1 is best unless loglikelihood is
                  slow (meaning taking many ms)
        args: extra positional arguments to pass to logprobability
        kwargs: extra keyword arguments to pass to logprobability
        """
        self.parameters = parameters
        self.num_walkers = num_walkers
        self.args = args
        self.kwargs = kwargs
        self.logprobability = logprobability
        self.jumping_distribution_set = jumping_distribution_set
        self.num_threads = num_threads
        self.reset()
    
    @property
    def args(self):
        """
        Property storing a sequence of positional arguments to pass to the
        logprobability function.
        """
        if not hasattr(self, '_args'):
            raise AttributeError("args was referenced before it was set.")
        return self._args
    
    @args.setter
    def args(self, value):
        """
        Setter for the positional arguments to pass to the logprobability
        function.
        
        value: must be a sequence
        """
        if type(value) in sequence_types:
            self._args = value
        else:
            raise TypeError("args was set to a non-sequence.")
    
    @property
    def kwargs(self):
        """
        Property storing a sequence of keyword arguments to pass to the
        logprobability function.
        """
        if not hasattr(self, '_kwargs'):
            raise AttributeError("kwargs was referenced before it was set.")
        return self._kwargs
    
    @kwargs.setter
    def kwargs(self, value):
        """
        Setter for the keyword arguments to pass to the logprobability function
        
        value: must be a dictionary with string keys
        """
        if isinstance(value, dict):
            if all([isinstance(key, basestring) for key in value.keys()]):
                self._kwargs = value
            else:
                raise TypeError("At least one of the keys of kwargs was " +\
                    "not a string.")
        else:
            raise TypeError("kwargs was set to a non-dictionary.")
    
    @property
    def logprobability(self):
        """
        Property storing the Callable that computes the log of the probability
        density at a given point in parameter space.
        """
        if not hasattr(self, '_logprobability'):
            raise AttributeError("logprobability was referenced before it " +\
                "was set.")
        return self._logprobability
    
    @logprobability.setter
    def logprobability(self, value):
        """
        Setter for the logprobability function.
        
        value: a callable object
        """
        if callable(value):
            self._logprobability = value
        else:
            raise TypeError("The given logprobability function is not " +\
                "callable.")
    
    @property
    def random_number_generator(self):
        """
        Property storing the random number generator to use. When this is first
        evaluated it is set to numpy.random.mtrand.RandomState().
        """
        if not hasattr(self, '_random_number_generator'):
            self._random_number_generator = np.random.mtrand.RandomState()
        return self._random_number_generator

    @property
    def random_state(self):
        """
        The state of the internal random number generator. It is the result of
        calling get_state() on a numpy.random.mtrand.RandomState object. You
        can try to set this property but be warned that if you do this and it
        fails, it will do so silently.
        """
        return self.random_number_generator.get_state()

    @random_state.setter
    def random_state(self, state):
        """
        Try to set the state of the random number generator but fail silently
        if it doesn't work.
        """
        try:
            self.random_number_generator.set_state(state)
        except:
            pass
    
    @property
    def chain(self):
        """
        Property storing the chain of positions in parameter space of the MCMC
        iterates.
        """
        if not hasattr(self, '_chain'):
            raise AttributeError("chain was referenced before it was set.")
        return self._chain
    
    @chain.setter
    def chain(self, value):
        """
        Setter for the chain positions of this MCMC.
        
        value: a 3D array whose first (third) axis length is num_walkers
               (num_parameters)
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if (len(value.shape) == 3):
                (num_walkers, nlinks, nparameters) = value.shape
                if num_walkers == self.num_walkers:
                    if nparameters == self.num_parameters:
                        self._chain = value
                    else:
                        raise ValueError("The length of the third axis of " +\
                            "chain was not equal to num_parameters.")
                else:
                    raise ValueError("The length of the first axis of " +\
                        "chain was not equal to num_walkers.")
            else:
                raise ValueError("chain was not a 3D array.")
        else:
            raise TypeError("chain was set to a non-sequence.")
        
    
    @property
    def lnprobability(self):
        """
        Property storing the log of the probability density of the positions of
        the chain links.
        """
        if not hasattr(self, '_lnprobability'):
            raise AttributeError("lnprobability was referenced before it " +\
                "was set.")
        return self._lnprobability
    
    @lnprobability.setter
    def lnprobability(self, value):
        """
        Setter for the lnprobability array.
        
        value: 2D numpy.ndarray of shape whose first element is num_walkers
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if (len(value.shape) == 2):
                (num_walkers, num_links) = value.shape
                if num_walkers == self.num_walkers:
                    self._lnprobability = value
                else:
                    raise ValueError("The length of the first axis of " +\
                        "lnprobability was not equal to num_walkers.")
            else:
                raise ValueError("lnprobability was not a 2D array.")
        else:
            raise TypeError("lnprobability was set to a non-sequence.")
    
    @property
    def num_iterations(self):
        """
        Property storing the number of iterations stored in the sampler.
        """
        if not hasattr(self, '_num_iterations'):
            raise AttributeError("num_iterations was referenced before it " +\
                "was set.")
        return self._num_iterations
    
    @num_iterations.setter
    def num_iterations(self, value):
        """
        Setter for the number of iterations through which this Sampler has run.
        
        value: non-negative integer
        """
        if type(value) in int_types:
            if value >= 0:
                self._num_iterations = value
            else:
                raise ValueError("num_iterations was set to a negative " +\
                    "integer.")
        else:
            raise TypeError("num_iterations was set to a non-integer.")
    
    @property
    def num_accepted(self):
        """
        Property storing the number of accepted steps for each walker.
        """
        if not hasattr(self, '_num_accepted'):
            raise AttributeError("num_accepted was referenced before it " +\
                "was set.")
        return self._num_accepted
    
    @num_accepted.setter
    def num_accepted(self, value):
        """
        Setter for the number of accepted steps.
        
        value: 1D numpy.ndarray object of length num_walkers
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if (len(value.shape) == 1):
                num_walkers = value.shape[0]
                if num_walkers == self.num_walkers:
                    self._num_accepted = value
                else:
                    raise ValueError("The length of num_accepted was not " +\
                        "equal to num_walkers.")
            else:
                raise ValueError("num_accepted was not a 1D array.")
        else:
            raise TypeError("num_accepted was set to a non-sequence.")
    
    @property
    def acceptance_fraction(self):
        """
        Property storing the acceptance fraction given the current state of the
        sampler.
        """
        return self.num_accepted / self.num_iterations
    
    @property
    def num_threads(self):
        """
        Property storing the number of threads to use in calculating log
        likelihood values.
        """
        if not hasattr(self, '_num_threads'):
            raise AttributeError("num_threads referenced before it was set.")
        return self._num_threads
    
    @num_threads.setter
    def num_threads(self, value):
        """
        Setter for the number of threads to use in log likelihood calculations.
        
        value: a positive integer; 1 is best unless loglikelihood is very slow
        """
        if type(value) in int_types:
            if value > 0:
                self._num_threads = value
            else:
                raise ValueError("num_threads must be non-negative.")
        else:
            raise TypeError("num_threads was set to a non-int.")
    
    def create_pool(self):
        """
        Property storing the Pool object which is used for log likelihood
        calculations. It stores an object which has a map function and a close
        function.
        """
        if have_multiprocess and (self.num_threads > 1):
            return Pool(self.num_threads)
        else:
            if self.num_threads > 1:
                print("Python module 'multiprocess' is not installed, so " +\
                    "multithreading is not possible in this Sampler. " +\
                    "Either install multiprocess or set num_threads to 1.")
            return DummyPool()
    
    @property
    def num_walkers(self):
        """
        Property storing the integer number of independent MCMC iterates
        evolved by this sampler.
        """
        if not hasattr(self, '_num_walkers'):
            raise AttributeError("num_walkers referenced before it was set.")
        return self._num_walkers
    
    @num_walkers.setter
    def num_walkers(self, value):
        """
        Setter for the number of independent MCMC iterates evolved by this
        sampler.
        
        value: a positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._num_walkers = value
            else:
                raise ValueError("num_walkers property must be a positive " +\
                    "integer.")
        else:
            raise TypeError("num_walkers was set to a non-int.")
    
    @property
    def parameters(self):
        """
        Property storing a list of the string names of the parameters explored
        by this sampler.
        """
        if not hasattr(self, '_parameters'):
            raise AttributeError("parameters referenced before it was set.")
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        """
        Setter for the names of the parameters explored by this sampler.
        
        value: sequence of strings
        """
        if type(value) in [list, tuple, np.ndarray]:
            if all([isinstance(element, basestring) for element in value]):
                self._parameters = [element for element in value]
            else:
                raise TypeError("At least one parameter given was not a " +\
                    "string.")
        else:
            raise TypeError("parameters was set to a non-sequence.")
    
    @property
    def num_parameters(self):
        """
        Property storing the integer number of parameters (i.e. the dimension
        of the explored space).
        """
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = len(self.parameters)
        return self._num_parameters
    
    @property
    def jumping_distribution_set(self):
        """
        Property storing the JumpingDistributionSet object storing the proposal
        distributions of this MetropolisHastingsSampler.
        """
        if not hasattr(self, '_jumping_distribution_set'):
            raise AttributeError("jumping_distribution_set referenced " +\
                "before it was set.")
        return self._jumping_distribution_set
    
    @jumping_distribution_set.setter
    def jumping_distribution_set(self, value):
        """
        Setter for the JumpingDistributionSet object storing the proposal
        distributions of this MetropolisHastingsSampler.
        
        value: JumpingDistributionSet object storing proposal distributions for
               the string parameters being explored by this sampler
        """
        if isinstance(value, JumpingDistributionSet):
            if set(self.parameters) == set(value.params):
                self._jumping_distribution_set = value
            else:
                raise ValueError("The given jumping_distribution_set did " +\
                    "not contain exactly the same parameters as the given " +\
                    "list of parameters given to the initializer.")
        else:
            raise TypeError("jumping_distribution_set was set to something " +\
                "which is not a JumpingDistributionSet object.")

    def reset(self):
        """
        Clear chain, lnprobability, and the bookkeeping parameters.
        """
        self._chain = np.empty((self.num_walkers, 0, self.num_parameters))
        self.lnprobability = np.empty((self.num_walkers, 0))
        self.num_iterations = 0
        self.num_accepted = np.zeros((self.num_walkers,), dtype=int)
        self._last_run_mcmc_result = None
    
    def sample(self, point, lnprob=None, random_state=None, num_iterations=1):
        """
        Advances the chain iterations steps as an iterator

        point: the initial position vector(s).
        lnprob: the logprobability evaluated at position point
                If lnprob is not provided, the initial value is calculated.
        random_state: (optional) the state of the random number generator. See
                     the :func:`random_state` property for details.
        num_iterations: (optional) integer number of steps to run.

        At each iteration, this generator yields (pos, lnprob, rstate) where:
                pos: the current positions of the chain in the parameter space
                lnprob: the value of the log posterior at pos
                rstate: the current state of the random number generator
        """
        self.random_state = random_state
        point = np.array(point)
        if type(lnprob) is type(None):
            lnprob = np.array([self.logprobability(point[iwalker], *self.args,\
                **self.kwargs) for iwalker in range(self.num_walkers)])
        self._chain = np.concatenate((self._chain, np.zeros(\
            (self.num_walkers, num_iterations, self.num_parameters))), axis=1)
        self.lnprobability = np.concatenate((self.lnprobability,\
            np.zeros((self.num_walkers, num_iterations))), axis=1)
        for i in range(num_iterations):
            randoms = [np.random.RandomState(\
                seed=self.random_number_generator.randint(2 ** 32))\
                for iwalker in range(self.num_walkers)]
            pool = self.create_pool()
            arguments = [(point[iwalker], lnprob[iwalker],\
                self.jumping_distribution_set, self.logprobability, self.args,\
                self.kwargs, self.parameters, randoms[iwalker])\
                for iwalker in range(self.num_walkers)]
            (destinations, newlnprobs, accepted, randoms) = zip(*pool.map(\
                choose_destination, arguments))
            pool.close()
            point = np.array(destinations)
            lnprob = np.array(newlnprobs)
            self.num_accepted =\
                self.num_accepted + np.array(accepted).astype(int)
            self._chain[:,self.num_iterations,:] = point
            self.lnprobability[:,self.num_iterations] = lnprob
            self.num_iterations = self.num_iterations + 1
            yield point, lnprob, self.random_state
    
    def run_mcmc(self, initial_position, num_iterations,\
        initial_random_state=None, initial_lnprob=None, **kwargs):
        """
        Iterate sample function for specified number of iterations.
        
        initial_position: initial position vector
        num_iterations: number of steps to run
        initial_lnprob: log posterior probability density at initial_position.
                        If initial_lnprob is not provided, the initial value is
                        calculated.
        initial_random_state: state of the random number generator. See the
                              random_state property for details.
        kwargs: keyword arguments that are passed to the sample function
        """
        for results in self.sample(initial_position, initial_lnprob,\
            initial_random_state, num_iterations=num_iterations, **kwargs):
            pass
        return results

