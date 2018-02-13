"""
File: distpy/Loadable.py
Author: Keith Tauscher
Date: 12 August 2017

Description: File containing subclass of any object which can be saved in an
             hdf5 file because it has a function with signature
             savable.fill_hdf5_group(group).
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
    Class representing an object which can be loaded from an hdf5 group or file
    directly through the class. This is distinct from the Savable class because
    there are some (specifically, self-nest-able) objects which cannot be
    loaded conveniently through methods of this form. These objects require an
    outside method of the form "load_XXXXX_from_hdf5_group" functions.
    """
    @staticmethod
    def load_from_hdf5_group(group):
        """
        A function which loads an instance of the current Savable subclass from
        the given hdf5 file group. This function raises an error unless it is
        implemented by all subclasses of Savable.
        
        group: hdf5 file group from which to load an instance of a Savable
             subclass
        """
        raise NotImplementedError("This method should be implemented by " +\
            "every subclass of Savable and Savable should never be " +\
            "instantiated directly.")
    
    @classmethod
    def load(cls, file_name, *args, **kwargs):
        """
        Loads an instance of a subclass of the Savable class. This method
        raises an error unless load_from_hdf5_group is defined in the given
        Savable subclass.
        
        file_name: the name of the file from which to load an instance of a
                   subclass of the Savable class
        
        returns: an instance of the class which called this method
        """
        if have_h5py:
            hdf5_file = h5py.File(file_name, 'r')
            to_return = cls.load_from_hdf5_group(hdf5_file, *args, **kwargs)
            hdf5_file.close()
            return to_return
        else:
            raise no_h5py_error

