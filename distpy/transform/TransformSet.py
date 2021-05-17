"""
Module containing class representing an unordered set of
`distpy.transform.Transform.Transform` objects indexed by string parameter
names.

**File**: $DISTPY/distpy/transform/TransformSet.py  
**Author**: Keith Tauscher  
**Date**: 15 May 2021
"""
from ..util import sequence_types, Savable, Loadable
from .NullTransform import NullTransform
from .CastTransform import cast_to_transform, castable_to_transform
from .LoadTransform import load_transform_from_hdf5_group
from .InvertTransform import invert_transform
from .TransformList import TransformList

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class TransformSet(Savable, Loadable):
    """
    Class representing an unordered set of
    `distpy.transform.Transform.Transform` objects indexed by string
    parameter names.
    """
    def __init__(self, transforms, parameters=None):
        """
        Initializes a new `TransformSet` object with the given
        `distpy.transform.Transform.Transform` objects and  string parameter
        names.
        
        Parameters
        ----------
        transforms : dict or sequence
            either a dictionary of `distpy.transform.Transform.Transform`
            objects indexed by strings or a sequence of
            `distpy.transform.Transform.Transform` objects
        parameters : sequence or None
            sequence of string parameter names to which sequence of transforms
            apply. `parameters` is only required (and only used) if
            `transforms` is a sequence as opposed to a dictionary
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
            if type(parameters) is type(None):
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
        The dictionary of `distpy.transform.Transform.Transform` objects
        indexed by string keys that stores the data of this container.
        """
        if not hasattr(self, '_transforms'):
            raise AttributeError("transforms referenced before it was set.")
        return self._transforms
    
    @property
    def inverse(self):
        """
        A `TransformSet` with the same keys and inverse transformations.
        """
        if not hasattr(self, '_inverse'):
            self._inverse = TransformSet(\
                {key: invert_transform(self.transforms[key])\
                for key in self.transforms})
        return self._inverse
    
    def subset(self, parameters):
        """
        Creates and returns another `TransformSet` object corresponding to
        the `distpy.transform.Transform.Transform` objects in this
        `TransformSet` that are indexed by the given parameters.
        
        Parameters
        ----------
        parameters : set
            a set of string parameter names describing the keys to include in
            the returned subset
        
        Returns
        -------
        transform_subset : `TransformSet`
            subset of this `TransformSet` object
        """
        if isinstance(parameters, set):
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
        else:
            raise TypeError("parameters given to subset was not a set.")
    
    def transform_list(self, parameters):
        """
        Creates and returns a `distpy.transform.TransformList.TransformList`
        object corresponding to the transforms in this `TransformSet` that are
        indexed by the given parameters.
        
        Parameters
        ----------
        parameters : sequence
            sequence of strings which are unique keys of transforms in this
            `TransformSet` in the order that they should be included in the
            returned `distpy.transform.TransformList.TransformList` object
        
        Returns
        -------
        transform_list : `distpy.transform.TransformList.TransformList`
            list form of this `TransformSet`
        """
        if type(parameters) in sequence_types:
            if all([isinstance(element, basestring)\
                for element in parameters]):
                if all([(element in self.transforms)\
                    for element in parameters]):
                    new_transforms =\
                        [self.transforms[element] for element in parameters]
                    return TransformList(*new_transforms)
                else:
                    raise KeyError("At least one of the strings given was " +\
                        "not a key for a transform in this TransformSet.")
            else:
                raise TypeError("Not all elements of the given sequence " +\
                    "were strings.")
        else:
            raise TypeError("parameters given to transform_list was not a " +\
                "sequence.")
    
    def __getitem__(self, key):
        """
        Gets the `distpy.transform.Transform.Transform` associated with the
        given key.
        
        Parameters
        ----------
        key : str or set or sequence
            a string parameter associated with the desired
            `distpy.transform.Transform.Transform` object, an unordered set of
            such keys, or a sequence of such keys
        
        Returns
        -------
        value : `distpy.transform.Transform.Transform` or `TransformSet` or\
        `distpy.transform.TransformList.TransformList`
            `value` depends on the type of `key`
            
            - if `key` is a string, `value` is the
            `distpy.transform.Transform.Transform` stored under parameter name
            given by `key`
            - if `key` is a set, `value` is a `TransformSet` containing only
            the `distpy.transform.Transform.Transform` objects associated with
            the parameter names in `key`
            - if `key` is a sequence (list, tuple, etc.), `value` is a
            `distpy.transform.TransformList.TransformList` containing the
            `distpy.transform.Transform.Transform` objects associated with the
            parameter names given in `key` (in the same order)
        """
        if isinstance(key, set):
            return self.subset(key)
        elif type(key) in sequence_types:
            return self.transform_list(key)
        elif key in self.transforms:
            return self.transforms[key]
        else:
            raise KeyError(("{!s} not in the transforms dictionary at the " +\
                "heart of this TransformSet object.").format(key))
    
    def __iter__(self):
        """
        Finds an iterator over this `TransformSet`, which is the same as the
        iterator of the internal `TransformSet.transforms` dictionary. The
        iterator will return a new string key each iteration.
        
        Returns
        -------
        iterator : dict_keyiterator
            iterator returned by the underlying dictionary of
            `distpy.transform.Transform.Transform` objects (indexed by string
            parameter names)
        """
        return self.transforms.__iter__()
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this
        `TransformSet`.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group in which to save information about this
            `TransformSet`
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
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group which previously had a `TransformSet` object saved
            in it
        
        Returns
        -------
        transform_set : `TransformSet`
            `transform_set` as loaded from the given hdf5 file group
        """
        subgroup = group['transforms']
        transforms = {par: load_transform_from_hdf5_group(subgroup[par])\
            for par in subgroup}
        return TransformSet(transforms)
    
    def apply(self, untransformed_parameters):
        """
        Applies the `distpy.transform.Transform.Transform` objects in this
        `TransformSet` to the given untransformed parameter values.
        
        Parameters
        ----------
        untransformed_parameters : dict
            a dictionary of parameter values (in the untransformed space) with
            the same string keys as this `TransformSet`
        
        Returns
        -------
        transformed_parameters : dict
            a dictionary of parameter values (in the transformed space) with
            the same string keys as this `TransformSet`
        """
        return {par: self[par](untransformed_parameters[par]) for par in self}
    
    def __call__(self, untransformed_parameters):
        """
        Applies the `distpy.transform.Transform.Transform` objects in this
        `TransformSet` to the given untransformed parameter values. This method
        simply calls the `TransformSet.apply` method.
        
        Parameters
        ----------
        untransformed_parameters : dict
            a dictionary of parameter values (in the untransformed space) with
            the same string keys as this `TransformSet`
        
        Returns
        -------
        transformed_parameters : dict
            a dictionary of parameter values (in the transformed space) with
            the same string keys as this `TransformSet`
        """
        return self.apply(untransformed_parameters)
    
    def apply_inverse(self, transformed_parameters):
        """
        Applies the inverses of the `distpy.transform.Transform.Transform`
        objects in this `TransformSet` to the given parameters values (given in
        the transformed space).
        
        Parameters
        ----------
        transformed_parameters : dict
            a dictionary of parameter values (in the transformed space) with
            the same string keys as this `TransformSet`
        
        Returns
        -------
        untransformed_parameters : dict
            a dictionary of parameter values (in the untransformed space) with
            the same string keys as this `TransformSet`
        """
        return {par: self[par].I(transformed_parameters[par]) for par in self}
    
    def I(self, transformed_parameters):
        """
        Applies the inverses of the `distpy.transform.Transform.Transform`
        objects in this `TransformSet` to the given parameters values (given in
        the transformed space). This method simply calls the
        `TransformSet.apply_inverse` method.
        
        Parameters
        ----------
        transformed_parameters : dict
            a dictionary of parameter values (in the transformed space) with
            the same string keys as this `TransformSet`
        
        Returns
        -------
        untransformed_parameters : dict
            a dictionary of parameter values (in the untransformed space) with
            the same string keys as this `TransformSet`
        """
        return self.apply_inverse(transformed_parameters)
    
    def __contains__(self, key):
        """
        Checks if the given key has a `distpy.transform.Transform.Transform`
        object associated with it in this `TransformSet`.
        
        Parameters
        ----------
        key : str
            name of parameter to check for
        
        Returns
        -------
        check_value : bool
            True if there is a `distpy.transform.Transform.Transform object in
            this `TransformSet` associated with `key`
        """
        return (key in self.transforms)
    
    def __len__(self):
        """
        Finds the number of `distpy.transform.Transform.Transform` objects
        contained in this `TransformSet` object.
        
        Returns
        -------
        length : int
            integer number of `distpy.transform.Transform.Transform` objects
            stored in this `TransformSet`
        """
        return len(self.transforms)
    
    @property
    def is_null(self):
        """
        Boolean describing whether this `TransformSet` encodes the
        `len(self)`-length null transformation.
        """
        for parameter in self.transforms:
            if not isinstance(self.transforms[parameter], NullTransform):
                return False
        return True
    
    def __eq__(self, other):
        """
        Checks to see if `self` and `other` are the same.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `self` and `other` encode the same
            `distpy.transform.Transform.Transform` objects and string
            parameters
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
        Checks to see if `self` and `other` are different.
        
        Parameters
        ----------
        other : object
            object to check for inequality
        
        Returns
        -------
        result : bool
            False if and only if `self` and `other` encode the same
            `distpy.transform.Transform.Transform` objects and string
            parameters
        """
        return (not self.__eq__(other))

