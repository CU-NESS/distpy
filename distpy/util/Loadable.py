"""
Module containing abstract class of any object which can be loaded from an hdf5
file via a method with signature `loadable.load_from_hdf5_group(group)`.

**File**: $DISTPY/distpy/util/Loadable.py  
**Author**: Keith Tauscher  
**Date**: 15 May 2021
"""
try:
    import h5py
except:
    have_h5py = False
    no_h5py_error = NotImplementedError("Saving couldn't be completed " +\
        "because h5py was not installed.")
else:
    have_h5py = True

class Loadable(object):
    """
    Abstract class that can be implemented by any object which can be loaded
    from an hdf5 file via a staticmethod with signature
    `loadable.load_from_hdf5_group(group)`.
    """
    @staticmethod
    def load_from_hdf5_group(group):
        """
        A function which loads an instance of the current
        `distpy.util.Loadable.Loadable` subclass from the given hdf5 file
        group. This function raises an error unless it is implemented by the
        subclass of `distpy.util.Loadable.Loadable` that is calling it.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group from which to load an instance of a
            `distpy.util.Loadable.Loadable` subclass
        
        Returns
        -------
        obj : subclass of `distpy.util.Loadable.Loadable`
            an instance of the class which called this method
        """
        raise NotImplementedError("This method should be implemented by " +\
            "every subclass of Loadable and Loadable should never be " +\
            "instantiated directly.")
    
    @classmethod
    def load(cls, file_name, *args, **kwargs):
        """
        Class method that loads an instance of a subclass of the
        `distpy.util.Loadable.Loadable` class. This method raises an error
        `distpy.util.Loadable.Loadable.load_from_hdf5_group` is defined in the
        given `distpy.util.Loadable.Loadable` subclass. Raises an error if
        `h5py` cannot be imported.
        
        Parameters
        ----------
        file_name : str
            the name of the file from which to load an instance of a subclass
            of the Savable class
        args : sequence
            positional arguments to pass to the
            `distpy.util.Loadable.Loadable.load_from_hdf5_group` method of the
            given subclass of `distpy.util.Loadable.Loadable`
        kwargs : dict
            keyword arguments to pass to the
            `distpy.util.Loadable.Loadable.load_from_hdf5_group` method of the
            given subclass of `distpy.util.Loadable.Loadable`
        
        Returns
        -------
        obj : subclass of `distpy.util.Loadable.Loadable`
            an instance of the class which called this method
        """
        if have_h5py:
            hdf5_file = h5py.File(file_name, 'r')
            to_return = cls.load_from_hdf5_group(hdf5_file, *args, **kwargs)
            hdf5_file.close()
            return to_return
        else:
            raise no_h5py_error

