"""
Module containing abstract class of any object which can be saved in an hdf5
file via a function with signature `savable.fill_hdf5_group(group)`.

**File**: $DISTPY/distpy/util/Savable.py  
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

class Savable(object):
    """
    Abstract class of any object which can be saved in an hdf5 file via a
    function with signature `savable.fill_hdf5_group(group)`.
    """
    def fill_hdf5_group(self, group):
        """
        A function which fills the given hdf5 file group with information about
        this `distpy.util.Savable.Savable` object. This function raises an
        error unless it is implemented by the subclass of
        `distpy.util.Savable.Savable` that is calling it.
        
        Parameters
        ----------
        group: h5py.Group
            hdf5 file group to fill with information about this object
        """
        raise NotImplementedError("This method should be implemented by " +\
            "every subclass of Savable and Savable should never be " +\
            "instantiated directly.")
    
    def save(self, file_name, *args, **kwargs):
        """
        Saves this object in hdf5 file using the
        `distpy.util.Savable.Savable.fill_hdf5_group` function. Raises an error
        if `h5py` cannot be imported.
        
        Parameters
        ----------
        file_name : str
            name of hdf5 file to write
        args : sequence
            positional arguments to pass to the
            `distpy.util.Savable.Savable.fill_hdf5_group` method of the given
            subclass of `distpy.util.Savable.Savable`
        kwargs : dict
            keyword arguments to pass to the
            `distpy.util.Savable.Savable.fill_hdf5_group` method of the given
            subclass of `distpy.util.Savable.Savable`
        """
        if have_h5py:
            hdf5_file = h5py.File(file_name, 'w')
            self.fill_hdf5_group(hdf5_file, *args, **kwargs)
            hdf5_file.close()
        else:
            raise no_h5py_error

