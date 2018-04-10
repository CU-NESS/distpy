"""
File: distpy/jumping/MetropolisHastingsSampler.py
Author: Keith Tauscher (with comment contribution from DFM's emcee)
Date: 12 Feb 2018

Description: File containing class implementing the abstract Sampler class from
             emcee using distpy's JumpingDistributionSet objects for the
             storing of proposal distributions.
"""
import numpy as np
from emcee import Sampler as emceeSampler
from ..util import int_types
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

class MetropolisHastingsSampler(emceeSampler):
    """
    Class implementing the abstract Sampler class from emcee using distpy's
    JumpingDistributionSet objects for the storing of proposal distributions.
    """
    def __init__(self, parameters, nwalkers, logprobability,\
        jumping_distribution_set, nthreads=1, args=[], kwargs={}):
        """
        Initializes a new sampler of the given log probability function.
        
        parameters: names of parameters explored by this sampler
        nwalkers: the number of independent MCMC iterates should be run at once
        logprobability: callable taking parameter array as input
        jumping_distribution_set: JumpingDistributionSet object storing
                                  proposal distributions used to sample given
                                  log probability function
        nthreads: the number of threads to use in log likelihood calculations
                  for walkers. Default: 1, 1 is best unless loglikelihood is
                  very slow
        args: extra positional arguments to pass to logprobability
        kwargs: extra keyword arguments to pass to logprobability
        """
        self.parameters = parameters
        self.nwalkers = nwalkers
        super(MetropolisHastingsSampler, self).__init__(len(self.parameters),\
            logprobability, *args, **kwargs)
        self.jumping_distribution_set = jumping_distribution_set
        self.nthreads = nthreads
    
    @property
    def nthreads(self):
        """
        Property storing the number of threads to use in calculating log
        likelihood values.
        """
        if not hasattr(self, '_nthreads'):
            raise AttributeError("nthreads referenced before it was set.")
        return self._nthreads
    
    @nthreads.setter
    def nthreads(self, value):
        """
        Setter for the number of threads to use in log likelihood calculations.
        
        value: a positive integer; 1 is best unless loglikelihood is very slow
        """
        if type(value) in int_types:
            if value > 0:
                self._nthreads = value
            else:
                raise ValueError("nthreads must be non-negative.")
        else:
            raise TypeError("nthreads was set to a non-int.")
    
    def create_pool(self):
        """
        Property storing the Pool object which is used for log likelihood
        calculations. It stores an object which has a map function and a close
        function.
        """
        if have_multiprocess and (self.nthreads > 1):
            return Pool(self.nthreads)
        else:
            if self.nthreads > 1:
                print("Python module 'multiprocess' is not installed, so " +\
                    "multithreading is not possible in this Sampler. " +\
                    "Either install multiprocess or set nthreads to 1.")
            return DummyPool()
    
    @property
    def nwalkers(self):
        """
        Property storing the integer number of independent MCMC iterates
        evolved by this sampler.
        """
        if not hasattr(self, '_nwalkers'):
            raise AttributeError("nwalkers referenced before it was set.")
        return self._nwalkers
    
    @nwalkers.setter
    def nwalkers(self, value):
        """
        Setter for the number of independent MCMC iterates evolved by this
        sampler.
        
        value: a positive integer
        """
        if isinstance(value, int):
            if value > 0:
                self._nwalkers = value
            else:
                raise ValueError("nwalkers property must be a positive " +\
                    "integer.")
        else:
            raise TypeError("nwalkers was set to a non-int.")
    
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
    
    def array_from_dictionary(self, dictionary):
        """
        Creates an array out of the dictionary.
        
        dictionary: dict with parameters as keys and arrays or numbers as
                    values 
        
        returns: array of ndim 1 greater than values ndim
        """
        return\
            np.array([dictionary[parameter] for parameter in self.parameters])
    
    def dictionary_from_array(self, array):
        """
        Creates a dictionary out of the arrays.
        
        array: numpy.ndarray whose first dimension has length num_parameters
        
        returns: dictionary with parameters as keys
        """
        return {parameter: array[iparameter]\
            for (iparameter, parameter) in enumerate(self.parameters)}

    def reset(self):
        """
        Clear chain, lnprobability, and the bookkeeping parameters.
        """
        self._chain = np.empty((self.nwalkers, 0, self.dim))
        self._lnprob = np.empty((self.nwalkers, 0))
        self.iterations = 0
        self.naccepted = np.zeros((self.nwalkers,), dtype=int)
        self._last_run_mcmc_result = None
    
    def sample(self, point, lnprob=None, randomstate=None, thin=1,\
        storechain=True, iterations=1):
        """
        Advances the chain ``iterations`` steps as an iterator

        point: the initial position vector.
        lnprob: (optional) the log posterior probability at position ``point``.
                If lnprob is not provided, the initial value is calculated.
        rstate0: (optional) the state of the random number generator. See the
                 :func:`random_state` property for details.
        iterations: (optional) the number of steps to run.
        thin: (optional) if you only want to store and yield every thin samples
              in the chain, set thin to an integer greater than 1.
        storechain: (optional) by default, the sampler stores (in memory) the
                    positions and log-probabilities of the samples in the
                    chain. If you are using another method to store the samples
                    to a file or if you don't need to analyse the samples after
                    the fact (for burn-in for example) set storechain to False.

        At each iteration, this generator yields (pos, lnprob, rstate) where:
                pos: the current positions of the chain in the paramete space
                lnprob: the value of the log posterior at pos
                rstate: the current state of the random number generator
        """
        self.random_state = randomstate
        point = np.array(point)
        if lnprob is None:
            lnprob = np.array([self.get_lnprob(point[iwalker])\
                for iwalker in range(self.nwalkers)])
        # Resize the chain in advance.
        if storechain:
            nlinks = int(iterations / thin)
            chain_shape = (self.nwalkers, nlinks, self.num_parameters)
            self._chain =\
                np.concatenate((self._chain, np.zeros(chain_shape)), axis=1)
            likelihood_shape = (self.nwalkers, nlinks)
            self._lnprob = np.concatenate((self._lnprob,\
                np.zeros(likelihood_shape)), axis=1)
        i0 = self.iterations
        for i in range(int(iterations)):
            self.iterations += 1
            destination_dicts = []
            destinations = []
            for iwalker in range(self.nwalkers):
                # Calculate the proposal distribution.
                source_dict = self.dictionary_from_array(point[iwalker])
                destination_dict =\
                    self.jumping_distribution_set.draw(source_dict)
                destination = self.array_from_dictionary(destination_dict)
                destination_dicts.append(destination_dict)
                destinations.append(destination)
            pool = self.create_pool()
            newlnprobs = list(pool.map(self.get_lnprob, destinations))
            pool.close()
            for iwalker in range(self.nwalkers):
                #newlnprob = self.get_lnprob(destination)
                newlnprob = newlnprobs[iwalker]
                log_value_difference =\
                    self.jumping_distribution_set.log_value_difference(\
                    source_dict, destination_dicts[iwalker])
                diff = newlnprob - lnprob[iwalker] - log_value_difference
                # M-H acceptance ratio
                if diff < 0:
                    diff = np.exp(diff) - self._random.rand()
                if diff > 0:
                    point[iwalker,:] = destinations[iwalker]
                    lnprob[iwalker] = newlnprob
                    self.naccepted[iwalker] += 1
            if storechain and (i % thin) == 0:
                ind = i0 + int(i / thin)
                self._chain[:,ind,:] = point
                self._lnprob[:,ind] = lnprob
            yield point, lnprob, self.random_state

