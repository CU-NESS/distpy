"""
File: distpy/transform/TransformSet.py
Author: Keith Tauscher
Date: 22 Feb 2018

Description: File containing a class representing an unordered set of Transform
             objects indexed by string parameters.
"""
from ..util import sequence_types, Savable, Loadable
from .CastTransform import cast_to_transform, castable_to_transform
from .LoadTransform import load_transform_from_hdf5_group
from .TransformList import TransformList

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class TransformSet(Savable, Loadable):
    """
    Class representing an unordered set of Transform objects indexed by string
    parameters.
    """
    def __init__(self, transforms, parameters=None):
        """
        Initializes a new TransformSet object with the given transforms and
        parameters.
        
        transforms: either a dictionary of Transform objects indexed by strings
                    or a sequence of Transform objects
        parameters: sequence of parameters to which sequence of transforms
                    apply. parameters is only required (and only used) if
                    transforms is a sequence as opposed to a dictionary
        """
        if isinstance(transforms, dict):
            if any([(not isinstance(key, basestring)) for key in transforms]):
                raise TypeError("All keys of transforms dictionary must be " +\
                    "strings.")
            if any([(not castable_to_transform(transforms[key]))\
                for key in transforms]):
                raise TypeError("Not all values of the given transforms " +\
                    "dictionary could be cast to Transform objects.")
            self._transforms =\
                {parameter: cast_to_transform(transforms[parameter])\
                for parameter in transforms}
        elif type(transforms) in sequence_types:
            if parameters is None:
                raise ValueError("parameters must be a sequence if " +\
                    "transforms is a sequence.")
            elif type(parameters) not in sequence_types:
                raise TypeError("parameters should be a sequence of strings.")
            if any([(not isinstance(par, basestring)) for par in parameters]):
                raise TypeError(\
                    "All elements of parameters should be strings.")
            if any([(not castable_to_transform(tfm)) for tfm in transforms]):
                raise TypeError(\
                    "All elements of transforms should be Transform objects.")
            if len(parameters) != len(transforms):
                raise ValueError("The number of string parameters must be " +\
                    "equal to the number of Transform objects.")
            self._transforms = {parameter: cast_to_transform(transform)\
                for (parameter, transform) in zip(parameters, transforms)}
    
    @property
    def transforms(self):
        """
        Property storing a dictionary of Transform objects indexed by string
        keys.
        """
        if not hasattr(self, '_transforms'):
            raise AttributeError("transforms referenced before it was set.")
        return self._transforms
    
    def __getitem__(self, key):
        """
        Gets the transform associated with the given key.
        
        key: a string parameter associated with the desired Transform object,
             an unordered set of such keys, or a sequence of such keys
        
        returns: if key is a string, Transform stored under given string key
                 if key is a set, TransformSet containing only the transforms
                                  associated with the strings in key
                 if key is a sequence (list, tuple, etc.), a TransformList
                                                           containing the
                                                           transforms
                                                           associated with the
                                                           given strings in the
                                                           order given in key
        """
        if isinstance(key, set):
            if all([isinstance(element, basestring) for element in key]):
                if all([(element in self.transforms) for element in key]):
                    new_transforms =\
                        {element: self.transforms[element] for element in key}
                    return TransformSet(new_transforms)
                else:
                    raise KeyError("At least one of the strings given was " +\
                        "not a key for a transform in this TransformSet.")
            else:
                raise KeyError("Not all elements of the given set were " +\
                    "strings.")
        elif type(key) in sequence_types:
            if all([isinstance(element, basestring) for element in key]):
                if all([(element in self.transforms) for element in key]):
                    new_transforms =\
                        [self.transforms[element] for element in key]
                    return TransformList(*new_transforms)
                else:
                    raise KeyError("At least one of the strings given was " +\
                        "not a key for a transform in this TransformSet.")
            else:
                raise TypeError("Not all elements of the given sequence " +\
                    "were strings.")
        elif key in self.transforms:
            return self.transforms[key]
        else:
            raise KeyError(("{!s} not in the transforms dictionary at the " +\
                "heart of this TransformSet object.").format(key))
    
    def __iter__(self):
        """
        Returns an iterator over this TransformSet, which is the same as the
        iterator of the internal transforms dictionary. The iterator will
        return a new string key each iteration.
        """
        return self.transforms.__iter__()
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this
        TransformSet.
        
        group: hdf5 file group in which to save information about this
               TransformSet
        """
        subgroup = group.create_group('transforms')
        for parameter in self.transforms:
            transform = self.transforms[parameter]
            subsubgroup = subgroup.create_group(parameter)
            transform.fill_hdf5_group(subsubgroup)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a TransformSet object from the given hdf5 file group.
        
        group: hdf5 file group which previously had a TransformSet object saved
               in it
        
        returns: a TransformSet object loaded from the given hdf5 file group
        """
        subgroup = group['transforms']
        transforms = {par: load_transform_from_hdf5_group(subgroup[par])\
            for par in subgroup}
        return TransformSet(transforms)
    
    def apply(self, untransformed_parameters):
        """
        Applies the Transform objects in this TransformSet to the given
        untransformed parameter values.
        
        untransformed_parameters: a dictionary of parameter values (in the
                                  untransformed space) with the same keys as
                                  this TransformSet
        
        returns: a dictionary of transformed parameter values
        """
        return {par: self[par](untransformed_parameters[par]) for par in self}
    
    def __call__(self, untransformed_parameters):
        """
        Applies the Transform objects in this TransformSet to the given
        untransformed parameter values. Equivalent to apply function.
        
        untransformed_parameters: a dictionary of parameter values (in the
                                  untransformed space) with the same keys as
                                  this TransformSet
        
        returns: a dictionary of transformed parameter values
        """
        return self.apply(parameters)
    
    def apply_inverse(self, transformed_parameters):
        """
        Applies the inverses of the Transform objects in this TransformSet to
        the given parameters values (given in the transformed space)
        
        transformed_parameters: a dictionary of parameter values (in the
                                transformed space) with the same keys as this
                                TransformSet
        
        returns: a dictionary of untransformed parameter values
        """
        return {par: self[par].I(transformed_parameters[par]) for par in self}
    
    def I(self, transformed_parameters):
        """
        Applies the inverses of the Transform objects in this TransformSet to
        the given parameters values (given in the transformed space)
        
        transformed_parameters: a dictionary of parameter values (in the
                                transformed space) with the same keys as this
                                TransformSet
        
        returns: a dictionary of untransformed parameter values
        """
        return self.apply_inverse(transformed_parameters)
    
    def __contains__(self, key):
        """
        Checks if the given key has a Transform object in this TransformSet
        associated with it.
        
        key: string parameter value
        
        returns: True if there is a Transform object in this TransformSet
                 associated with the given key
        """
        return (key in self.transforms)
    
    def __len__(self):
        """
        Finds the number of Transform objects contained in this TransformSet
        object.
        
        returns: integer number of Transform objects stored in this
                 TransformSet
        """
        return len(self.transforms)
    
    def __eq__(self, other):
        """
        Checks to see if self and other are equal or unequal
        
        other: object to check for equality
        
        returns: True if self and other encode the same Transform objects and
                      string parameters
                 False otherwise
        """
        if not isinstance(other, TransformSet):
            return False
        if len(self) != len(other):
            return False
        for parameter in self:
            if parameter in other:
                if self[parameter] != other[parameter]:
                    return False
            else:
                return False
        return True
    
    def __ne__(self, other):
        """
        Checks to see if self and other are unequal or equal.
        
        other: object to check for inequality
        
        returns: False if self and other encode the same Transform objects and
                       string parameters
                 True otherwise
        """
        return (not self.__eq__(other))

