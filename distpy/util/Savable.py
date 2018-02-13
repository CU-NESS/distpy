"""
File: distpy/Savable.py
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

class Savable(object):
    """
    Class representing an object which can be saved in an hdf5 file group
    because it has a function with signature savable.fill_hdf5_group(group).
    """
    def fill_hdf5_group(self, group):
        """
        A function which fills the given hdf5 file group with information about
        this Savable object. This function raises an error unless it is
        implemented by all subclasses of Savable.
        
        group: hdf5 file group to fill with information about this object
        """
        raise NotImplementedError("This method should be implemented by " +\
            "every subclass of Savable and Savable should never be " +\
            "instantiated directly.")
    
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
    
    def save(self, file_name, *args, **kwargs):
        """
        Saves this object in hdf5 file using the fill_hdf5_file group
        function. Raises an error if h5py cannot be imported.
        
        file_name: name of hdf5 file to write
        """
        if have_h5py:
            hdf5_file = h5py.File(file_name, 'w')
            self.fill_hdf5_group(hdf5_file, *args, **kwargs)
            hdf5_file.close()
        else:
            raise no_h5py_error

