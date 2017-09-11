"""
File: distpy/Saving.py
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
    def save(self, file_name):
        """
        Saves DistributionSet in hdf5 file using the fill_hdf5_file group
        function. Raises an error if h5py cannot be imported.
        
        file_name: name of hdf5 file to write
        """
        if have_h5py:
            hdf5_file = h5py.File(file_name, 'w')
            self.fill_hdf5_group(hdf5_file)
            hdf5_file.close()
        else:
            raise no_h5py_error

