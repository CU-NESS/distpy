"""
Module containing class that performs a Metropolis Hastings (MH) Markov Chain
Monte Carlo (MCMC) sampling, which essentially samples a difficult distribution
by sampling a simpler distribution and rejecting points so that the final
distribution matches the target distribution. It is a Markov Chain because the
distribution of each point in the sampling depends only on the location of the
last drawn point.

**File**: $DISTPY/distpy/jumping/MetropolisHastingsSampler.py  
**Author**: Keith Tauscher  
**Date**: 11 Jul 2021
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
    Creates an array out of the given parameter dictionary.
    
    Parameters
    ----------
    dictionary : dict
        dictionary with parameter names as keys and arrays or numbers as values
    
    Returns
    -------
    array : numpy.ndarray
        array with one more dimension than that of the values of dictionary
        containing the parameter values from the dictionary
    """
    return np.array([dictionary[parameter] for parameter in parameters])

def dictionary_from_array(array, parameters):
    """
    Creates a dictionary out of the arrays, assuming they represent values of
    the given parameters.
    
    Parameters
    ----------
    array : numpy.ndarray
        array whose first dimension has length `len(parameters)`
    parameters : sequence
        sequence of string parameter names to use as keys of the returned
        dictionary
    
    Returns
    -------
    dictionary : dict
        dictionary with parameters as keys and numbers from `array` as values
    """
    return {parameter: array[iparameter]\
        for (iparameter, parameter) in enumerate(parameters)}

def choose_destination(input_tuple):
    """
    Chooses a destination point given the source point and its loglikelihood.
    
    Parameters
    ----------
    input_tuple : tuple
        tuple of the form `(source, lnprob, jumping_distribution_set,\
        logprobability, args, kwargs, parameters, random)`:
        
        - `source` is a 1D `numpy.ndarray` describing the source point
        - `lnprob` is the log of the target PDF evaluated at `source`
        - `jumping_distribution_set` is a
        `distpy.jumping.JumpingDistribution.JumpingDistribution` object
        describing how to evaluate and draw jumps
        - `logprobability` is a Callable that gives the value of the target PDF
        at a given point
        - `args` is a sequence of positional arguments to pass to
        `logprobability`
        - `kwargs` is a dictionary of keyword arguments to pass to
        `logprobability`
        - `parameters` is a sequence of string parameter names
        - `random` the random number generator in a specific state
    
    Returns
    -------
    output_tuple : tuple
        tuple of the form `(destination, newlnprob, accepted, random)`:
        
        - `destination` is a 1D `numpy.ndarray` describing the destination
        point
        - `newlnprob` is the log probability evaluated at `destination`
        - `accepted` is a boolean describing whether or not the jump performed
        in this function was accepted to arrive at `destination`
        - `random` is the random number generator in an updated state
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
    Class representing a dummy-`Pool` which uses the built-in map function to
    do its mapping. This will be used if `multiprocess` is not installed or if
    `MetropolisHastingsSampler.num_threads` is set to 1.
    """
    def map(self, function, iterable):
        """
        Calls the given function on each element of the given iterable.
        
        Parameters
        ----------
        function : Callable
            function to apply to each element of iterable
        iterable : sequence
            elements to which to apply function
        
        Returns
        -------
        mapped : sequence
            sequence whose elements are found by applying the function to the
            corresponding element of the iterable
        """
        return map(function, iterable)
    
    def close(self):
        """
        Closes the pool (by doing nothing).
        """
        pass

class MetropolisHastingsSampler(object):
    """
    Class that performs a Metropolis Hastings (MH) Markov Chain Monte Carlo
    (MCMC) sampling, which essentially samples a difficult distribution by
    sampling a simpler distribution and rejecting points so that the final
    distribution matches the target distribution. It is a Markov Chain because
    the distribution of each point in the sampling depends only on the location
    of the last drawn point.
    """
    def __init__(self, parameters, num_walkers, logprobability,\
        jumping_distribution_set, num_threads=1, args=[], kwargs={}):
        """
        Initializes a new sampler of the given log probability function.
        
        Parameters
        ----------
        parameters : sequence
            sequence of string names of parameters explored by this sampler
        num_walkers : int
            the number of independent MCMC iterates should be run at once
        logprobability : Callable
            Callable taking parameter array as input
        jumping_distribution_set : `distpy.jumping.JumpingDistributionSet.JumpingDistributionSet`
            object storing proposal distributions used to sample given log
            probability function
        num_threads : int
            the number of threads to use in log likelihood calculations for
            walkers. Default: 1, 1 is best unless loglikelihood is slow
            (meaning taking many ms)
        args : sequence
            extra positional arguments to pass to `logprobability`
        kwargs : dict
            extra keyword arguments to pass to `logprobability`
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
        A sequence of positional arguments to pass to
        `MetropolisHastingsSampler.logprobability`.
        """
        if not hasattr(self, '_args'):
            raise AttributeError("args was referenced before it was set.")
        return self._args
    
    @args.setter
    def args(self, value):
        """
        Setter for `MetropolisHastingsSampler.args`
        
        Parameters
        ----------
        value : sequence
            sequence of positional arguments to pass to
            `MetropolisHastingsSampler.logprobability`
        """
        if type(value) in sequence_types:
            self._args = value
        else:
            raise TypeError("args was set to a non-sequence.")
    
    @property
    def kwargs(self):
        """
        A dictionary of keyword arguments to pass to
        `MetropolisHastingsSampler.logprobability`.
        """
        if not hasattr(self, '_kwargs'):
            raise AttributeError("kwargs was referenced before it was set.")
        return self._kwargs
    
    @kwargs.setter
    def kwargs(self, value):
        """
        Setter for `MetropolisHastingsSampler.kwargs`
        
        Parameters
        ----------
        value : dict
            dictionary of keyword arguments to pass to
            `MetropolisHastingsSampler.logprobability`
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
        The Callable that computes the log of the target probability density at
        a given point in parameter space.
        """
        if not hasattr(self, '_logprobability'):
            raise AttributeError("logprobability was referenced before it " +\
                "was set.")
        return self._logprobability
    
    @logprobability.setter
    def logprobability(self, value):
        """
        Setter for `MetropolisHastingsSampler.logprobability`.
        
        Parameters
        ----------
        value : Callable
            an object that can be called to evaluate the log of the target
            probability density
        """
        if callable(value):
            self._logprobability = value
        else:
            raise TypeError("The given logprobability function is not " +\
                "callable.")
    
    @property
    def random_number_generator(self):
        """
        The random number generator to use. When this is first evaluated it is
        set to `numpy.random.mtrand.RandomState()`.
        """
        if not hasattr(self, '_random_number_generator'):
            self._random_number_generator = np.random.mtrand.RandomState()
        return self._random_number_generator

    @property
    def random_state(self):
        """
        The state of the internal random number generator. It is the result of
        calling `get_state()` on a `numpy.random.mtrand.RandomState` object.
        You can try to set this property but be warned that if you do this and
        it fails, it will do so silently.
        """
        return self.random_number_generator.get_state()

    @random_state.setter
    def random_state(self, state):
        """
        Sets `MetropolisHastingsSampler.random_state`. This is done internally
        and the user should only set this at their own risk.
        
        Parameters
        ----------
        state : tuple
            value like the one returned by
            `numpy.random.mtrand.RandomState.get_state`
        """
        try:
            self.random_number_generator.set_state(state)
        except:
            pass
    
    @property
    def chain(self):
        """
        The chain of positions in parameter space of the MCMC iterates. It is a
        `numpy.ndarray` of shape `(MetropolisHastingsSampler.num_walkers,\
        num_steps, MetropolisHastingsSampler.num_parameters)`
        """
        if not hasattr(self, '_chain'):
            raise AttributeError("chain was referenced before it was set.")
        return self._chain
    
    @chain.setter
    def chain(self, value):
        """
        Setter for `MetropolisHastingsSampler.chain`.
        
        Parameters
        ----------
        value : numpy.ndarray
            a 3D array of shape `(MetropolisHastingsSampler.num_walkers,\
            num_steps, MetropolisHastingsSampler.num_parameters)`
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
        The log of the probability density of the positions of the chain links.
        It is a 2D array of shape
        `(MetropolisHastingsSampler.num_walkers, num_steps)`
        """
        if not hasattr(self, '_lnprobability'):
            raise AttributeError("lnprobability was referenced before it " +\
                "was set.")
        return self._lnprobability
    
    @lnprobability.setter
    def lnprobability(self, value):
        """
        Setter for `MetropolisHastingsSampler.lnprobability`.
        
        Parameters
        ----------
        value : numpy.ndarray
            2D array of shape
            `(MetropolisHastingsSampler.num_walkers, num_steps)`
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
        The integer number of iterations stored in the sampler.
        """
        if not hasattr(self, '_num_iterations'):
            raise AttributeError("num_iterations was referenced before it " +\
                "was set.")
        return self._num_iterations
    
    @num_iterations.setter
    def num_iterations(self, value):
        """
        Setter for `MetropolisHastingsSampler.num_iterations`.
        
        Parameters
        ----------
        value : int
            non-negative integer number of iterations performed
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
        The number of accepted steps for each walker as a 1D array of length
        `MetropolisHastingsSampler.num_walkers`.
        """
        if not hasattr(self, '_num_accepted'):
            raise AttributeError("num_accepted was referenced before it " +\
                "was set.")
        return self._num_accepted
    
    @num_accepted.setter
    def num_accepted(self, value):
        """
        Setter for `MetropolisHastingsSampler.num_accepted`.
        
        Parameters
        ----------
        value : numpy.ndarray
            1D array of length `MetropolisHastingsSampler.num_walkers`
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
        The acceptance fraction given the current state of the sampler as a 1D
        numpy.ndarray of length `MetropolisHastingsSampler.num_walkers` whose
        values are between 0 and 1 (inclusive).
        """
        return self.num_accepted / self.num_iterations
    
    @property
    def num_threads(self):
        """
        The integer number of threads to use in calculating log likelihood
        values.
        """
        if not hasattr(self, '_num_threads'):
            raise AttributeError("num_threads referenced before it was set.")
        return self._num_threads
    
    @num_threads.setter
    def num_threads(self, value):
        """
        Setter for `MetropolisHastingsSampler.num_threads`.
        
        Parameters
        ----------
        value : int
            a positive integer. `value=1` is best unless loglikelihood is very
            slow
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
        The `multiprocess.Pool` or `DummyPool` object which is used for log
        likelihood calculations. It stores an object which has a
        `map(function, iterable)` function and a `close()` function.
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
        The integer number of independent MCMC iterates evolved by this
        sampler.
        """
        if not hasattr(self, '_num_walkers'):
            raise AttributeError("num_walkers referenced before it was set.")
        return self._num_walkers
    
    @num_walkers.setter
    def num_walkers(self, value):
        """
        Setter for `MetropolisHastingsSampler.num_walkers`.
        
        Parameters
        ----------
        value : int
            a positive integer number of walkers to use. Using a number similar
            in magnitude to the number of parameters is typically best
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
        A list of the string names of the parameters explored by this sampler.
        """
        if not hasattr(self, '_parameters'):
            raise AttributeError("parameters referenced before it was set.")
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        """
        Setter for `MetropolisHastingsSampler.parameters`.
        
        Parameters
        ----------
        value : sequence
            sequence of string parameter names
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
        The integer number of parameters (i.e. the dimension of the explored
        space).
        """
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = len(self.parameters)
        return self._num_parameters
    
    @property
    def jumping_distribution_set(self):
        """
        The `distpy.jumping.JumpingDistributionSet.JumpingDistributionSet`
        object storing the proposal distributions of this
        `MetropolisHastingsSampler`.
        """
        if not hasattr(self, '_jumping_distribution_set'):
            raise AttributeError("jumping_distribution_set referenced " +\
                "before it was set.")
        return self._jumping_distribution_set
    
    @jumping_distribution_set.setter
    def jumping_distribution_set(self, value):
        """
        Setter for `MetropolisHastingsSampler.jumping_distribution_set`.
        
        Parameters
        ----------
        value : `distpy.jumping.JumpingDistributionSet.JumpingDistributionSet`
            object storing proposal distributions for the string parameters
            being explored by this sampler
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
        Clears `MetropolisHastingsSampler.chain`,
        `MetropolisHastingsSampler.lnprobability`,
        `MetropolisHastingsSampler.num_iterations`, and
        `MetropolisHastingsSampler.num_accepted` to reset the sampler.
        """
        self._chain = np.empty((self.num_walkers, 0, self.num_parameters))
        self.lnprobability = np.empty((self.num_walkers, 0))
        self.num_iterations = 0
        self.num_accepted = np.zeros((self.num_walkers,), dtype=int)
        self._last_run_mcmc_result = None
    
    def sample(self, point, lnprob=None, random_state=None, num_iterations=1):
        """
        Advances the chain `num_iterations` steps as an iterator.
        
        Parameters
        ----------
        point : numpy.ndarray
            the initial position vectors of the walkers as a 2D array of shape
            `(MetropolisHastingsSampler.num_walkers,\
            MetropolisHastingsSampler.num_parameters)`.
        lnprob : numpy.ndarray or None
            the value of the logprobability function at the given `point`
            
            - if `lnprob` is an array, it should be 1D and have length
            `MetropolisHastingsSampler.num_walkers`
            - if `lnprob` is None, the log probability is calculated by this
            method.
        random_state : tuple
            (optional) the state of the random number generator. See
            `MetropolisHastingsSampler.random_state`.
        num_iterations : int
            integer number of steps to run. defaults to 1
        
        Returns
        -------
        state_tuple : tuple
            At each iteration, this generator yields
            `state_tuple=(pos, lnprob, rstate)` where:
            
            - `pos` contains the current positions of the chain in the
            parameter space in an array of the same shape as the `points`
            parameter
            - `lnprob`: the value of the log posterior at `pos`, stored in an
            array of the same shape as the `lnprob` parameter
            - `rstate` is the current state of the random number generator,
            stored as a tuple
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
        Iterates `MetropolisHastingsSampler.sample` for specified number of
        iterations.
        
        Parameters
        ----------
        initial_position : numpy.ndarray
            the initial position vectors of the walkers as a 2D array of shape
            `(MetropolisHastingsSampler.num_walkers,\
            MetropolisHastingsSampler.num_parameters)`.
        num_iterations : int
            number of steps to run
        initial_lnprob : numpy.ndarray or None
            the value of the logprobability function at the given `point`
            
            - if `lnprob` is an array, it should be 1D and have length
            `MetropolisHastingsSampler.num_walkers`
            - if `lnprob` is None, the log probability is calculated by this
            method.
        initial_random_state : tuple
            (optional) the state of the random number generator. See
            `MetropolisHastingsSampler.random_state`.
        kwargs : dict
            keyword arguments that are passed to the sample function
        """
        for results in self.sample(initial_position, initial_lnprob,\
            initial_random_state, num_iterations=num_iterations, **kwargs):
            pass
        return results

