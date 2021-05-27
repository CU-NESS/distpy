"""
Module containing class representing a sparse matrix, which is block diagonal,
where all blocks are square, i.e. a matrix of the form: $$\\boldsymbol{M} =\
\\begin{bmatrix} \\boldsymbol{A}_1 & \\boldsymbol{0} & \\cdots &\
\\boldsymbol{0} \\\\ \\boldsymbol{0} & \\boldsymbol{A}_2 & \\cdots &\
\\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\\
\\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N\
\\end{bmatrix}$$ An efficiency boost can be achieved if the
blocks are all the same shape.

**File**: $DISTPY/distpy/util/SparseSquareBlockDiagonalMatrix.py  
**Author**: Keith Tauscher  
**Date**: 18 May 2021
"""
from __future__ import division
import numpy as np
import numpy.linalg as npla
import scipy.linalg as scila
from scipy import sparse
from .TypeCategories import int_types, real_numerical_types, sequence_types
from .h5py_extensions import create_hdf5_dataset, get_hdf5_value
from .Savable import Savable
from .Loadable import Loadable

class SparseSquareBlockDiagonalMatrix(Savable, Loadable):
    """
    Class representing a sparse matrix, which is block diagonal, where all
    blocks are square, i.e. a matrix of the form:
    $$\\boldsymbol{M} = \\begin{bmatrix} \\boldsymbol{A}_1 & \\boldsymbol{0} &\
    \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} & \\boldsymbol{A}_2 &\
    \\cdots & \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\\
    \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N\
    \\end{bmatrix},$$ An efficiency boost can be achieved if
    the blocks are all the same shape.
    """
    def __init__(self, blocks, symmetry_tolerance=1e-12):
        """
        Initializes a new `SparseSquareBlockDiagonalMatrix`, which
        represents a matrix of the form: $$\\boldsymbol{M} = \\begin{bmatrix}\
        \\boldsymbol{A}_1 & \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\\
        \\boldsymbol{0} & \\boldsymbol{A}_2 & \\cdots & \\boldsymbol{0} \\\\\
        \\vdots & \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
        \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N \\end{bmatrix},$$
        
        Parameters
        ----------
        blocks : sequence
            sequence of square matrices, \\(\\boldsymbol{A}_1,\
            \\boldsymbol{A}_2, \\ldots, \\boldsymbol{A}_N\\) as `numpy.ndarray`
            objects. If all blocks are the same size, many operations will be
            more efficient. In this case, the most efficient way to supply
            `blocks` is as a 3-dimensional `numpy.ndarray` indexed such that
            `blocks[i,j,k]` describes the element in the \\(j^{\\text{th}}\\)
            row of the \\(k^{\\text{th}}\\) column of the \\(i^{\\text{th}}\\)
            block.
        """
        self.blocks = blocks
        self.symmetry_tolerance = symmetry_tolerance
        # this will allow for __rmatmul__ to implement left matrix
        # multiplication in the future, when numpy adds __matmul__ as a ufunc
        # (in the future).
        self.__array_ufunc__ = None
    
    @staticmethod
    def concatenate(*matrices):
        """
        Creates a new `SparseSquareBlockDiagonalMatrix` object by making the
        given matrices blocks in a new, larger block diagonal matrix.
        
        Parameters
        ----------
        matrices : sequence
            a sequence of `SparseSquareBlockDiagonalMatrix` objects
        
        Returns
        -------
        concatenation : `SparseSquareBlockDiagonalMatrix`
            if `matrices` are represented by \\(\\left\\{ \\begin{bmatrix}\
            \\boldsymbol{A}^{(1)}_1 & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} \\\\ \\boldsymbol{0} & \\boldsymbol{A}^{(1)}_2 &\
            \\cdots & \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots &\
            \\vdots \\\\ \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{A}^{(1)}_{N_1} \\end{bmatrix}, \\begin{bmatrix}\
            \\boldsymbol{A}^{(2)}_1 & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} \\\\ \\boldsymbol{0} & \\boldsymbol{A}^{(2)}_2 &\
            \\cdots & \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots &\
            \\vdots \\\\ \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{A}^{(2)}_{N_2} \\end{bmatrix}, \\ldots,\
            \\begin{bmatrix} \\boldsymbol{A}^{(M)}_1 & \\boldsymbol{0} &\
            \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            \\boldsymbol{A}^{(M)}_2 & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{A}^{(M)}_{N_M}\
            \\end{bmatrix} \\right\\},\\) then `concatenation` represents:
            $$\\begin{bmatrix} \\boldsymbol{A}^{(1)}_1 & \\boldsymbol{0} &\
            \\cdots & \\boldsymbol{0} & \\boldsymbol{0} & \\boldsymbol{0} &\
            \\cdots & \\boldsymbol{0} & \\cdots & \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            \\boldsymbol{A}^{(1)}_2 & \\cdots & \\boldsymbol{0} &\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{0} &\
            \\cdots & \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots & \\vdots &\
            \\vdots & \\vdots & \\ddots & \\vdots & \\ddots & \\vdots &\
            \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{A}^{(1)}_{N_1} &\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{0} &\
            \\cdots & \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} \\\\ \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} & \\boldsymbol{A}^{(2)}_1 & \\boldsymbol{0} &\
            \\cdots & \\boldsymbol{0} & \\cdots & \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} & \\boldsymbol{0} &\
            \\boldsymbol{A}^{(2)}_2 & \\cdots & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\\
            \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots &\
            \\ddots & \\vdots & \\ddots & \\vdots & \\vdots & \\ddots &\
            \\vdots \\\\ \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} & \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{A}^{(2)}_{N_2} & \\cdots & \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\ddots &\
            \\vdots & \\ddots & \\vdots & \\vdots & \\ddots & \\vdots \\\\\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{0} &\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{0} &\
            \\cdots & \\boldsymbol{A}^{(M)}_1 & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} \\\\ \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} & \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} &\
            \\boldsymbol{A}^{(M)}_2 & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\ddots &\
            \\vdots & \\ddots & \\vdots & \\vdots & \\ddots & \\vdots \\\\\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{0} &\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{0} &\
            \\cdots & \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{A}^{(M)}_{N_M} \\end{bmatrix}$$
        """
        if any([not isinstance(matrix, SparseSquareBlockDiagonalMatrix)\
            for matrix in matrices]):
            raise TypeError("All arguments passed to concatenate static " +\
                "method should be SparseSquareBlockDiagonalMatrix objects.")
        new_blocks = []
        for matrix in matrices:
            new_blocks.extend([block for block in matrix.blocks])
        return SparseSquareBlockDiagonalMatrix(new_blocks)
    
    @property
    def blocks(self):
        """
        The list of square blocks,
        \\(\\boldsymbol{A}_1,\\boldsymbol{A}_2,\\ldots,\\boldsymbol{A}_N\\)
        as `numpy.ndarray` objects.
        """
        if not hasattr(self, '_blocks'):
            raise AttributeError("blocks was referenced before it was set.")
        return self._blocks
    
    @blocks.setter
    def blocks(self, value):
        """
        Setter for `SparseSquareBlockDiagonalMatrix.blocks`.
        
        Parameters
        ----------
        value : sequence
            sequence of square matrices, \\(\\boldsymbol{A}_1,\
            \\boldsymbol{A}_2, \\ldots, \\boldsymbol{A}_N\\) as `numpy.ndarray`
            objects.
        """
        if type(value) in sequence_types:
            if isinstance(value, np.ndarray):
                if value.ndim == 3:
                    if value.shape[-2] == value.shape[-1]:
                        self._blocks = value.astype(float)
                        self._efficient = True
                        self._block_sizes = [value.shape[-1]] * value.shape[0]
                    else:
                        raise ValueError("blocks was set to a " +\
                            "numpy.ndarray that implies that the blocks " +\
                            "are not square.")
                else:
                    raise TypeError("blocks was set to a numpy.ndarray " +\
                        "with a different number of dimensions ({:d}) than " +\
                        "3.".format(value.ndim))
            else:
                (new_value, sizes, efficient) = ([], [], True)
                for element in value:
                    if type(element) in real_numerical_types:
                        new_element = np.array([[element]]).astype(float)
                    elif type(element) in sequence_types:
                        new_element = np.array(element).astype(float)
                        if new_element.ndim != 2:
                            raise ValueError("At least one block of the " +\
                                "matrix was neither a number nor a 2D " +\
                                "numpy.ndarray.")
                        if new_element.shape[0] != new_element.shape[1]:
                            raise ValueError("At least one block of the " +\
                                "matrix was set to a non-square 2D " +\
                                "numpy.ndarray.")
                    else:
                        raise TypeError("At least one block of the matrix " +\
                            "was set to neither a number nor a sequence.")
                    new_size = len(new_element)
                    if len(sizes) > 0:
                        efficient = (efficient and (new_size == sizes[-1]))
                    sizes.append(new_size)
                    new_value.append(new_element)
                self._block_sizes = sizes
                self._blocks = new_value
                if efficient:
                    self._blocks = np.array(self._blocks)
                self._efficient = efficient
        else:
            raise TypeError("blocks was set to a non-sequence.")
    
    @property
    def num_blocks(self):
        """
        The integer number, \\(N\\), of blocks composing this matrix.
        """
        if not hasattr(self, '_num_blocks'):
            self._num_blocks = len(self.blocks)
        return self._num_blocks
    
    @property
    def block_sizes(self):
        """
        Sequence of integer sizes of the blocks on the diagonal,
        \\(\\text{dim}(\\boldsymbol{A}_1), \\text{dim}(\\boldsymbol{A}_2),\
        \\ldots, \\text{dim}(\\boldsymbol{A}_N)\\)
        """
        if not hasattr(self, '_block_sizes'):
            raise AttributeError("block_sizes was referenced before it was " +\
                "set.")
        return self._block_sizes
    
    @property
    def efficient(self):
        """
        Bool describing whether all blocks, \\(\\boldsymbol{A}_1,\
        \\boldsymbol{A}_2, \\ldots, \\boldsymbol{A}_N\\), have the same size.
        """
        if not hasattr(self, '_efficient'):
            raise AttributeError("efficient was referenced before it was set.")
        return self._efficient
    
    @property
    def dimension(self):
        """
        Integer dimension of the column (or row) space of this matrix,
        \\(\\text{dim}(\\boldsymbol{M})\\).
        """
        if not hasattr(self, '_dimension'):
            self._dimension = sum(self.block_sizes)
        return self._dimension
    
    @property
    def matrix(self):
        """
        This matrix, \\(\\boldsymbol{M}\\), in the form of a
        `scipy.sparse.spmatrix` object.
        """
        if not hasattr(self, '_matrix'):
            self._matrix = sparse.block_diag(self.blocks)
        return self._matrix
    
    def dense(self):
        """
        Returns a dense form of this matrix. WARNING: for large matrices, this
        may require an excessive amount of memory.
        
        Returns
        -------
        full_matrix : numpy.ndarray
            the matrix this object represents as a dense `numpy.ndarray`
        """
        return scila.block_diag(*self.blocks)
    
    @property
    def symmetry_tolerance(self):
        """
        The relative tolerance, \\(\\varepsilon\\), when checking whether this
        matrix is symmetric (many operations are faster for symmetric
        matrices). See `SparseSquareBlockDiagonalMatrix.symmetric` for more
        information.
        """
        if not hasattr(self, '_symmetry_tolerance'):
            raise AttributeError("symmetry_tolerance was referenced before " +\
                "it was set.")
        return self._symmetry_tolerance
    
    @symmetry_tolerance.setter
    def symmetry_tolerance(self, value):
        """
        Setter for `SparseSquareBlockDiagonalMatrix.symmetry_tolerance`
        
        Parameters
        ----------
        value : number
            any positive number.
        """
        if type(value) in real_numerical_types:
            if value >= 0:
                self._symmetry_tolerance = np.float64(value)
            else:
                raise ValueError("symmetry_tolerance was set to a negative " +\
                    "number.")
        else:
            raise TypeError("symmetry_tolerance was set to a non-number.")
    
    @property
    def symmetric(self):
        """
        Boolean describing whether this matrix is symmetric up to the
        `SparseSquareBlockDiagonalMatrix.symmetry_tolerance`,
        \\(\\varepsilon\\), which is
        \\(\\left|(\\boldsymbol{A}_k)_{ij}-(\\boldsymbol{A}_k)_{ji}\\right|\
        \\le\\varepsilon\\times\\left|(\\boldsymbol{A}_k)_{ij}\\right|,\\ \\ \
        \\forall k\\in\\{1,2,\\ldots,N\\}\\).
        """
        if not hasattr(self, '_symmetric'):
            if self.efficient:
                self._symmetric = np.allclose(self.blocks,\
                    np.swapaxes(self.blocks, -2, -1),\
                    rtol=self.symmetry_tolerance, atol=0)
            else:
                symmetric = True
                for block in self.blocks:
                    symmetric = (symmetric and np.allclose(block, block.T,\
                        rtol=self.symmetry_tolerance, atol=0))
                self._symmetric = symmetric
        return self._symmetric
    
    def _check_definiteness(self):
        """
        Checks if this matrix is positive-definite or positive-semidefinite.
        """
        if self.symmetric:
            if self.efficient:
                eigenvalues = npla.eigvalsh(self.blocks)
                self._positive_semidefinite = np.all(eigenvalues >= 0)
                self._positive_definite = np.all(eigenvalues > 0)
            else:
                (positive_semidefinite, positive_definite) = (True, True)
                for block in self.blocks:
                    if positive_semidefinite:
                        eigenvalues = npla.eigvalsh(block)
                        positive_semidefinite = np.all(eigenvalues >= 0)
                        if positive_definite:
                            positive_definite = np.all(eigenvalues > 0)
                self._positive_semidefinite = positive_semidefinite
                self._positive_definite = positive_definite
        else:
            self._positive_semidefinite = False
            self._positive_definite = False
    
    @property
    def positive_semidefinite(self):
        """
        Boolean describing whether \\(\\boldsymbol{M}\\) is symmetric (see
        `SparseSquareBlockDiagonalMatrix.symmetric`) and positive
        semi-definite, which means that
        \\(\\boldsymbol{v}^T\\boldsymbol{M}\\boldsymbol{v}\\ge 0\\) for all
        column vectors \\(\\boldsymbol{v}\\) satisfying
        \\(\\text{dim}(\\boldsymbol{v})=\\text{dim}(\\boldsymbol{M})\\).
        """
        if not hasattr(self, '_positive_semidefinite'):
            self._check_definiteness()
        return self._positive_semidefinite
    
    @property
    def positive_definite(self):
        """
        Boolean describing whether \\(\\boldsymbol{M}\\) is symmetric (see
        `SparseSquareBlockDiagonalMatrix.symmetric`) and positive-definite,
        which means that
        \\(\\boldsymbol{v}^T\\boldsymbol{M}\\boldsymbol{v}> 0\\) for all
        non-zero column vectors \\(\\boldsymbol{v}\\) satisfying
        \\(\\text{dim}(\\boldsymbol{v})=\\text{dim}(\\boldsymbol{M})\\).
        """
        if not hasattr(self, '_positive_definite'):
            self._check_definiteness()
        return self._positive_definite
    
    @staticmethod
    def combine_blocks(blocks1, blocks2, operation, **allclose_kwargs):
        """
        Combines the block lists using the given operation. Addition,
        subtraction, matrix multiplication, and equality checking are
        implemented (see `operation` argument).
        
        Parameters
        ----------
        blocks1 : sequence
            `SparseSquareBlockDiagonalMatrix.blocks` property for the first
            `SparseSquareBlockDiagonalMatrix` object
        blocks2 : sequence
            `SparseSquareBlockDiagonalMatrix.blocks` property for the second
            `SparseSquareBlockDiagonalMatrix` object
        operation : str
            one of '+' (for addition), '-' (for subtraction), '@' (for matrix
            multiplication), '==' (for equality check with `numpy.allclose`;
            see `allclose_kwargs` argument)
        allclose_kwargs : dict
            dictionary of keyword arguments to pass to `numpy.allclose`. Only
            used if `operation=='=='`.
        
        Returns
        -------
        result : bool or `SparseSquareBlockDiagonalMatrix`
            - if `operation=='=='`, a boolean is returned describing whether
            the matrices are equal to within the tolerances set in
            `allclose_kwargs`
            - for all other values of `operation`, the returned value is a new
            valid block list that can be used to create a new
            `SparseSquareBlockDiagonalMatrix`
        """
        supported_operations = ['+', '-', '@', '==']
        if operation not in supported_operations:
            raise NotImplementedError("The '{0!s}' operation is not on the " +\
                "supported_operations list, which is {1}.".format(\
                supported_operations))
        dim1 = sum([len(element) for element in blocks1])
        dim2 = sum([len(element) for element in blocks2])
        if dim1 != dim2:
            raise ValueError(("The two sets of blocks cannot be combined " +\
                "because they correspond to different size matrices ({0:d} " +\
                "and {1:d} for the first and second matrix, " +\
                "respectively).").format(dim1, dim2))
        (index1, index2) = (0, 0)
        (block1, block2) = (np.array([[]]), np.array([[]]))
        current = []
        while True:
            if block1.size < max(1, block2.size):
                if block1.size == 0:
                    block1 = blocks1[index1]
                else:
                    block1 = scila.block_diag(block1, blocks1[index1])
                index1 += 1
            elif block2.size < block1.size:
                if block2.size == 0:
                    block2 = blocks2[index2]
                else:
                    block2 = scila.block_diag(block2, blocks2[index2])
                index2 += 1
            if block1.size == block2.size:
                if operation == '+':
                    new_block = block1 + block2
                elif operation == '-':
                    new_block = block1 - block2
                elif operation == '@':
                    new_block = np.matmul(block1, block2)
                elif operation == '==':
                    new_block = np.allclose(block1, block2)
                else:
                    raise NotImplementedError(("The '{!s}' operation is " +\
                        "not implemented even though it was in the " +\
                        "supported_operations list. It should either be " +\
                        "implemented or removed from the " +\
                        "supported_operations list.").format(operation))
                current.append(new_block)
                (block1, block2) = (np.array([[]]), np.array([[]]))
                if (index1 == len(blocks1)) or (index2 == len(blocks2)):
                    break
        if operation == '==':
            return all(current)
        else:
            return current
    
    def __add__(self, other):
        """
        Implements matrix addition.
        
        Parameters
        ----------
        other : `SparseSquareBlockDiagonalMatrix`
            another `SparseSquareBlockDiagonalMatrix`
        
        Returns
        -------
        sum : `SparseSquareBlockDiagonalMatrix`
            an object whose dense matrix representation is equal to the sum of
            the dense representations of `self` and `other`. If `self` and
            `other` are represented by \\(\\boldsymbol{M}=\\begin{bmatrix}\
            \\boldsymbol{A}_1 & \\boldsymbol{0} & \\cdots & \\boldsymbol{0}\
            \\\\ \\boldsymbol{0} & \\boldsymbol{A}_2 & \\cdots &\
            \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N\
            \\end{bmatrix}\\) and \\(\\boldsymbol{N}=\\begin{bmatrix}\
            \\boldsymbol{B}_1 & \\boldsymbol{0} & \\cdots & \\boldsymbol{0}\
            \\\\ \\boldsymbol{0} & \\boldsymbol{B}_2 & \\cdots &\
            \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{B}_N\
            \\end{bmatrix}\\), respectively, and\
            \\(\\text{dim}(\\boldsymbol{A}_k)=\\text{dim}(\\boldsymbol{B}_k)\\)
            for all \\(k\\), then `sum` is:
            $$\\boldsymbol{M}+\\boldsymbol{N}=\\begin{bmatrix}\
            \\boldsymbol{A}_1 + \\boldsymbol{B}_1 & \\boldsymbol{0} &\
            \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            \\boldsymbol{A}_2+\\boldsymbol{B}_2 & \\cdots & \\boldsymbol{0}\
            \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N+\\boldsymbol{B}_N\
            \\end{bmatrix}$$
        """
        if not isinstance(other, SparseSquareBlockDiagonalMatrix):
            return NotImplemented
        if self.block_sizes == other.block_sizes:
            if self.efficient:
                new_blocks = self.blocks + other.blocks
            else:
                new_blocks = [(sblock + oblock)\
                    for (sblock, oblock) in zip(self.blocks, other.blocks)]
        else:
            new_blocks = SparseSquareBlockDiagonalMatrix.combine_blocks(\
                self.blocks, other.blocks, '+')
        return SparseSquareBlockDiagonalMatrix(new_blocks)
    
    def __neg__(self):
        """
        Finds \\(-\\boldsymbol{M}\\) by finding its individual blocks, i.e.
        \\(-\\boldsymbol{A}_1, -\\boldsymbol{A}_2, \\ldots,\
        -\\boldsymbol{A}_N\\).
        
        Returns
        -------
        result : `SparseSquareBlockDiagonalMatrix`
            block diagonal matrix composed of new blocks:
            $$-\\boldsymbol{M}=\\begin{bmatrix} -\\boldsymbol{A}_1 &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            -\\boldsymbol{A}_2 & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & -\\boldsymbol{A}_N \\end{bmatrix}$$
        """
        if self.efficient:
            new_blocks = (-self.blocks)
        else:
            new_blocks = [(-block) for block in self.blocks]
        return SparseSquareBlockDiagonalMatrix(new_blocks)
    
    def __sub__(self, other):
        """
        Implements matrix subtraction.
        
        Parameters
        ----------
        other : `SparseSquareBlockDiagonalMatrix`
            another `SparseSquareBlockDiagonalMatrix`
        
        Returns
        -------
        difference : `SparseSquareBlockDiagonalMatrix`
            an object whose dense matrix representation is equal to the sum of
            the dense representations of `self` and `other`. If `self` and
            `other` are represented by \\(\\boldsymbol{M}=\\begin{bmatrix}\
            \\boldsymbol{A}_1 & \\boldsymbol{0} & \\cdots & \\boldsymbol{0}\
            \\\\ \\boldsymbol{0} & \\boldsymbol{A}_2 & \\cdots &\
            \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N\
            \\end{bmatrix}\\) and \\(\\boldsymbol{N}=\\begin{bmatrix}\
            \\boldsymbol{B}_1 & \\boldsymbol{0} & \\cdots & \\boldsymbol{0}\
            \\\\ \\boldsymbol{0} & \\boldsymbol{B}_2 & \\cdots &\
            \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{B}_N\
            \\end{bmatrix}\\), respectively, and\
            \\(\\text{dim}(\\boldsymbol{A}_k)=\\text{dim}(\\boldsymbol{B}_k)\\)
            for all \\(k\\), then `difference` is:
            $$\\boldsymbol{M}-\\boldsymbol{N}=\\begin{bmatrix}\
            \\boldsymbol{A}_1 - \\boldsymbol{B}_1 & \\boldsymbol{0} &\
            \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            \\boldsymbol{A}_2-\\boldsymbol{B}_2 & \\cdots & \\boldsymbol{0}\
            \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N-\\boldsymbol{B}_N\
            \\end{bmatrix}$$
        """
        if not isinstance(other, SparseSquareBlockDiagonalMatrix):
            return NotImplemented
        if self.block_sizes == other.block_sizes:
            if self.efficient:
                new_blocks = self.blocks - other.blocks
            else:
                new_blocks = [(sblock - oblock)\
                    for (sblock, oblock) in zip(self.blocks, other.blocks)]
        else:
            new_blocks = SparseSquareBlockDiagonalMatrix.combine_blocks(\
                self.blocks, other.blocks, '-')
        return SparseSquareBlockDiagonalMatrix(new_blocks)
    
    def __mul__(self, other):
        """
        Multiplies this matrix by a scalar. For matrix multiplication for
        vectors or matrices, use the
        `SparseSquareBlockDiagonalMatrix.__matmul__` method.
        
        Parameters
        ----------
        other : number
            a scalar, \\(a\\), by which to multiply this matrix
        
        Returns
        -------
        product : `SparseSquareBlockDiagonalMatrix`
            product of \\(a\\) and this matrix, \\(\\boldsymbol{M}\\):
            $$a\\boldsymbol{M}=\\begin{bmatrix} a\\boldsymbol{A}_1 &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            a\\boldsymbol{A}_2 & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & a\\boldsymbol{A}_N \\end{bmatrix}$$
        """
        if type(other) not in real_numerical_types:
            return NotImplemented
        if self.efficient:
            new_blocks = self.blocks * other
        else:
            new_blocks = [(block * other) for block in self.blocks]
        return SparseSquareBlockDiagonalMatrix(new_blocks)
    
    def __div__(self, other):
        """
        Divides this matrix by a scalar.
        
        Parameters
        ----------
        other : number
            a (nonzero) scalar, \\(a\\), by which to divide this matrix
        
        Returns
        -------
        quotient : `SparseSquareBlockDiagonalMatrix`
            quotient of this matrix and \\(a\\), \\(\\boldsymbol{M}\\):
            $$a\\boldsymbol{M}=\\begin{bmatrix} \\boldsymbol{A}_1/a &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            \\boldsymbol{A}_2/a & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & a\\boldsymbol{A}_N/a \\end{bmatrix}$$
        """
        if type(other) not in real_numerical_types:
            return NotImplemented
        if other == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        if self.efficient:
            new_blocks = self.blocks / other
        else:
            new_blocks = [(block / other) for block in self.blocks]
        return SparseSquareBlockDiagonalMatrix(new_blocks)
    
    def __truediv__(self, other):
        """
        Divides this matrix by a scalar.
        
        Parameters
        ----------
        other : number
            a (nonzero) scalar, \\(a\\), by which to divide this matrix
        
        Returns
        -------
        quotient : `SparseSquareBlockDiagonalMatrix`
            quotient of this matrix and \\(a\\), \\(\\boldsymbol{M}\\):
            $$a\\boldsymbol{M}=\\begin{bmatrix} \\boldsymbol{A}_1/a &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            \\boldsymbol{A}_2/a & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & a\\boldsymbol{A}_N/a \\end{bmatrix}$$
        """
        return self.__div__(other)
    
    def __rmul__(self, other):
        """
        Multiplies this matrix by a scalar. For matrix multiplication for
        vectors or matrices, use the
        `SparseSquareBlockDiagonalMatrix.__matmul__` method.
        
        Parameters
        ----------
        other : number
            a scalar, \\(a\\), by which to multiply this matrix
        
        Returns
        -------
        product : `SparseSquareBlockDiagonalMatrix`
            product of \\(a\\) and this matrix, \\(\\boldsymbol{M}\\):
            $$a\\boldsymbol{M}=\\begin{bmatrix} a\\boldsymbol{A}_1 &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            a\\boldsymbol{A}_2 & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & a\\boldsymbol{A}_N \\end{bmatrix}$$
        """
        return self.__mul__(other)
    
    def array_matrix_multiplication(self, other, right=True):
        """
        Performs left or right multiplication of this matrix with a vector.
        
        Parameters
        ----------
        other : numpy.ndarray
            there are two cases where a `numpy.ndarray` input is acceptable:
            - 1D vectors of length
            `SparseSquareBlockDiagonalMatrix.dimension`
            - ND matrices of shape
            `(...,SparseSquareBlockDiagonalMatrix.dimension)` will be
            interpreted as a shaped list of 1D vectors as above
        right : bool
            True if multiplying \\(\\boldsymbol{M}\\boldsymbol{v}\\) and False
            if multiplying \\(\\boldsymbol{v}^T\\boldsymbol{M}\\)
        
        Returns
        -------
        product : numpy.ndarray
            - If `other` is a vector \\(\\begin{bmatrix} \\boldsymbol{v}_1\\\\\
            \\boldsymbol{v}_2 \\\\ \\vdots \\\\ \\boldsymbol{v}_N\
            \\end{bmatrix}\\) (or a list of such vectors), then `product` is a
            `numpy.ndarray` represented by
            \\(\\begin{bmatrix} \\boldsymbol{A}_1\\boldsymbol{v}_1 \\\\\
            \\boldsymbol{A}_2\\boldsymbol{v}_2 \\\\ \\vdots \\\\\
            \\boldsymbol{A}_N\\boldsymbol{v}_N \\end{bmatrix}\\) if `right` is
            True and \\(\\begin{bmatrix} \\boldsymbol{A}^T_1\\boldsymbol{v}_1\
            \\\\ \\boldsymbol{A}^T_2\\boldsymbol{v}_2 \\\\ \\vdots \\\\\
            \\boldsymbol{A}^T_N\\boldsymbol{v}_N \\end{bmatrix}\\). Note that
            blocking happens automatically by the method and does not need to
            be performed before input. `product` is guaranteed to have the same
            shape as `other`.
        """
        other = np.array(other)
        if other.ndim >= 1:
            original_shape = other.shape
        else:
            raise ValueError("other was set to a 0D array.")
        if other.shape[-1] != self.dimension:
            raise ValueError("The last dimension of other was not " +\
                "equal to the dimension of this " +\
                "SparseSquareBlockDiagonalMatrix.")
        # int required below because np.prod(()) is float(1), not int(1)
        num_vectors = int(np.prod(original_shape[:-1]))
        if self.efficient:
            other = np.reshape(other, (num_vectors, self.num_blocks, -1))
            if right:
                # self.blocks shape is (num_blocks, block_size, block_size)
                # other shape is (m,    num_blocks,             block_size)
                product = np.einsum('abc,dac->dab', self.blocks, other)
            else:
                # self.blocks shape is (num_blocks, block_size, block_size)
                # other shape is (m,    num_blocks, block_size)
                product = np.einsum('abc,dab->dac', self.blocks, other)
            # product shape is (m, num_blocks, block_size)
        else:
            other = np.reshape(other, (num_vectors, self.dimension))
            (product, accounted_for) = ([], 0)
            for (block_size, block) in zip(self.block_size, self.blocks):
                vblock = other[:,accounted_for:accounted_for+block_size]
                if right:
                    # block shape is  (   block_size, block_size)
                    # vblock shape is (m,             block_size)
                    product_block = np.matmul(vblock, block.T)
                    # is np.einsum('ab,cb->ac', vblock, block) faster? TODO
                else:
                    # block shape is  (    block_size, block_size)
                    # vblock shape is (m,  block_size)
                    product_block = np.matmul(vblock, block)
                product.append(product_block)
                accounted_for += block_size
            product = np.concatenate(product, axis=-1) 
            # product shape is (m, dimension)
        return np.reshape(product, original_shape)
    
    def __matmul__(self, other):
        """
        Multiplies this matrix by a vector or another matrix (note that this
        operation is not commutative) on the right.
        
        Parameters
        ----------
        other : numpy.ndarray or `SparseSquareBlockDiagonalMatrix`
            - there are two cases where a `numpy.ndarray` input is acceptable:
                - 1D vectors of length
                `SparseSquareBlockDiagonalMatrix.dimension`
                - ND matrices of shape
                `(...,SparseSquareBlockDiagonalMatrix.dimension)` will be
                interpreted as a shaped list of 1D vectors as above (to
                multiply this `SparseSquareBlockDiagonalMatrix` by another
                square matrix of the same size, it is recommended to pass
                `other` as another `SparseSquareBlockDiagonalMatrix`)
            - otherwise, `other` should be a `SparseSquareBlockDiagonalMatrix`
            to multiply this matrix by on the right
        
        Returns
        -------
        product : numpy.ndarray or `SparseSquareBlockDiagonalMatrix`
            - If `other` is a vector \\(\\begin{bmatrix} \\boldsymbol{v}_1\\\\\
            \\boldsymbol{v}_2 \\\\ \\vdots \\\\ \\boldsymbol{v}_N\
            \\end{bmatrix}\\) (or a list of such vectors), then `product` is a
            `numpy.ndarray` represented by
            \\(\\begin{bmatrix} \\boldsymbol{A}_1\\boldsymbol{v}_1 \\\\\
            \\boldsymbol{A}_2\\boldsymbol{v}_2 \\\\ \\vdots \\\\\
            \\boldsymbol{A}_N\\boldsymbol{v}_N \\end{bmatrix}\\). Note that
            blocking happens automatically by the method and does not need to
            be performed before input. `product` is guaranteed to have the same
            shape as `other`.
            - If `self` and `other` are both `SparseSquareBlockDiagonalMatrix`
            objects represented by \\(\\boldsymbol{M}=\\begin{bmatrix}\
            \\boldsymbol{A}_1 & \\boldsymbol{0} & \\cdots & \\boldsymbol{0}\
            \\\\ \\boldsymbol{0} & \\boldsymbol{A}_2 & \\cdots &\
            \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N\
            \\end{bmatrix}\\) and \\(\\boldsymbol{N}=\\begin{bmatrix}\
            \\boldsymbol{B}_1 & \\boldsymbol{0} & \\cdots & \\boldsymbol{0}\
            \\\\ \\boldsymbol{0} & \\boldsymbol{B}_2 & \\cdots &\
            \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\\
            \\boldsymbol{0} & \\boldsymbol{0} & \\cdots & \\boldsymbol{B}_N\
            \\end{bmatrix},\\) respectively, and\
            \\(\\text{dim}(\\boldsymbol{A}_k)=\\text{dim}(\\boldsymbol{B}_k)\\)
            for all \\(k\\), then `product` is a
            `SparseSquareBlockDiagonalMatrix` represented by:
            $$\\boldsymbol{M}\\boldsymbol{N}=\\begin{bmatrix}\
            \\boldsymbol{A}_1\\boldsymbol{B}_1 & \\boldsymbol{0} &\
            \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            \\boldsymbol{A}_2\\boldsymbol{B}_2 & \\cdots & \\boldsymbol{0}\
            \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N\\boldsymbol{B}_N\
            \\end{bmatrix}$$
        """
        if isinstance(other, SparseSquareBlockDiagonalMatrix):
            if self.block_sizes == other.block_sizes:
                if self.efficient:
                    new_blocks =\
                        np.einsum('ijk,ikl->ijl', self.blocks, other.blocks)
                else:
                    new_blocks = [np.matmul(sblock, oblock)\
                        for (sblock, oblock) in zip(self.blocks, other.blocks)]
            else:
                new_blocks = SparseSquareBlockDiagonalMatrix.combine_blocks(\
                    self.blocks, other.blocks, '@')
            return SparseSquareBlockDiagonalMatrix(new_blocks)
        elif type(other) in sequence_types:
            return self.array_matrix_multiplication(other, right=True)
        else:
            raise NotImplemented
    
    def __rmatmul__(self, other):
        """
        Multiplies this matrix by a vector or another matrix (note that this
        operation is not commutative) on the left. **NOTE**: This method does
        not get called by numpy because `matmul` is not considered a ufunc.
        Until the numpy developers make it one, use the
        `SparseSquareBlockDiagonalMatrix.array_matrix_multiplication` method
        with `right=False` to perform left matrix multiplication
        
        Parameters
        ----------
        other : numpy.ndarray or `SparseSquareBlockDiagonalMatrix`
            - there are two cases where a `numpy.ndarray` input is acceptable:
                - 1D vectors of length
                `SparseSquareBlockDiagonalMatrix.dimension`
                - ND matrices of shape
                `(...,SparseSquareBlockDiagonalMatrix.dimension)` will be
                interpreted as a shaped list of 1D vectors as above (to
                multiply this `SparseSquareBlockDiagonalMatrix` by another
                square matrix of the same size, it is recommended to pass
                `other` as another `SparseSquareBlockDiagonalMatrix`)
        
        Returns
        -------
        product : numpy.ndarray
            - If `other` is a vector \\(\\begin{bmatrix} \\boldsymbol{v}_1\\\\\
            \\boldsymbol{v}_2 \\\\ \\vdots \\\\ \\boldsymbol{v}_N\
            \\end{bmatrix}\\) (or a list of such vectors), then `product` is a
            `numpy.ndarray` represented by
            \\(\\begin{bmatrix} \\boldsymbol{A}_1^T\\boldsymbol{v}_1 \\\\\
            \\boldsymbol{A}^T_2\\boldsymbol{v}_2 \\\\ \\vdots \\\\\
            \\boldsymbol{A}^T_N\\boldsymbol{v}_N \\end{bmatrix}\\). Note that
            blocking happens automatically by the method and does not need to
            be performed before input. `product` is guaranteed to have the same
            shape as `other`.
        """
        if type(other) in sequence_types:
            return self.array_matrix_multiplication(other, right=False)
        else:
            raise NotImplemented
    
    def __pow__(self, exponent):
        """
        Finds \\(\\boldsymbol{M}^p\\) by finding its individual blocks, i.e.
        \\(\\boldsymbol{A}_1^p, \\boldsymbol{A}_2^p, \\ldots,\
        \\boldsymbol{A}_N^p\\).
        
        Parameters
        ----------
        exponent : real number
            number, \\(p\\), that can be positive, negative, or zero
        
        Returns
        -------
        inverse_matrix : `SparseSquareBlockDiagonalMatrix`
            block diagonal matrix composed of new blocks:
            $$\\boldsymbol{M}^p=\\begin{bmatrix} \\boldsymbol{A}_1^p &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            \\boldsymbol{A}_2^p & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N^p \\end{bmatrix}$$
        """
        if type(exponent) not in real_numerical_types:
            raise TypeError("SparseSquareBlockDiagonalMatrix can only be " +\
                "put to a real power.")
        integer_exponent =\
            (type(exponent) in int_types) or (exponent == int(exponent))
        if integer_exponent:
            if self.efficient:
                new_blocks = npla.matrix_power(self.blocks, int(exponent))
            else:
                new_blocks = [npla.matrix_power(block, int(exponent))\
                    for block in self.blocks]
        elif self.positive_semidefinite:
            if self.efficient:
                (eigenvalues, eigenvectors) = npla.eigh(self.blocks)
                eigenvalues = (eigenvalues ** exponent)[:,None,:]
                new_blocks = np.einsum('ijk,ilk->ijl',\
                    eigenvectors * eigenvalues, eigenvectors)
            else:
                new_blocks = []
                for block in self.blocks:
                    (eigenvalues, eigenvectors) = npla.eigh(block)
                    eigenvalues = (eigenvalues ** exponent)[None,:]
                    new_blocks.append(\
                        np.dot(eigenvectors * eigenvalues, eigenvectors.T))
        else:
            raise ValueError("Non-positive semi-definite matrices cannot " +\
                "be put to fractional powers because the result is not " +\
                "unique.")
        return SparseSquareBlockDiagonalMatrix(new_blocks)
    
    def __rpow__(self, base):
        """
        Finds \\(b^{\\boldsymbol{M}}\\) by finding its individual blocks, i.e.
        \\(b^{\\boldsymbol{A}_1}, b^{\\boldsymbol{A}_2}, \\ldots,\
        b^{\\boldsymbol{A}_N}\\).
        
        Parameters
        ----------
        exponent : real number
            \\(b\\), any positive number
        
        Returns
        -------
        inverse_matrix : `SparseSquareBlockDiagonalMatrix`
            block diagonal matrix composed of new blocks:
            $$b^{\\boldsymbol{M}}=\\begin{bmatrix} b^{\\boldsymbol{A}_1} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            b^{\\boldsymbol{A}_2} & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & b^{\\boldsymbol{A}_N} \\end{bmatrix}$$
        """
        if (type(base) not in real_numerical_types) or (base <= 0):
            raise TypeError("SparseSquareBlockDiagonalMatrix can only be " +\
                "put to a positive power.")
        if self.symmetric:
            if self.efficient:
                (eigenvalues, eigenvectors) = npla.eigh(self.blocks)
                eigenvalues = (base ** eigenvalues)[:,None,:]
                new_blocks = np.einsum('ijk,ilk->ijl',\
                    eigenvectors * eigenvalues, eigenvectors)
            else:
                new_blocks = []
                for block in self.blocks:
                    (eigenvalues, eigenvectors) = npla.eigh(block)
                    eigenvalues = (base ** eigenvalues)[None,:]
                    new_blocks.append(np.matmul(\
                        eigenvectors * eigenvalues, eigenvectors.T))
        elif self.efficient:
            (eigenvalues, eigenvectors) = npla.eig(self.blocks)
            eigenvalues = (base ** eigenvalues)[:,None,:]
            new_blocks = np.matmul(eigenvectors * eigenvalues,
                npla.inv(eigenvectors))
        else:
            new_blocks = []
            for block in self.blocks:
                (eigenvalues, eigenvectors) = npla.eig(block)
                eigenvalues = (base ** eigenvalues)[None,:]
                new_blocks.append(np.matmul(eigenvectors * eigenvalues,\
                    npla.inv(eigenvectors)))
        return SparseSquareBlockDiagonalMatrix(new_blocks)
    
    def square_root(self):
        """
        Finds \\(\\boldsymbol{M}^{1/2}\\) by finding the square roots of the
        individual blocks, i.e. finding \\(\\boldsymbol{A}_1^{1/2},\
        \\boldsymbol{A}_2^{1/2}, \\ldots, \\boldsymbol{A}_N^{1/2}\\). This can
        only be done when
        `SparseSquareBlockDiagonalMatrix.positive_semidefinite` is True.
        
        Returns
        -------
        square_root_matrix : `SparseSquareBlockDiagonalMatrix`
            block diagonal matrix composed of square root blocks:
            $$\\boldsymbol{M}^{1/2}=\\begin{bmatrix} \\boldsymbol{A}_1^{1/2} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            \\boldsymbol{A}_2^{1/2} & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N^{1/2}\
            \\end{bmatrix}$$
        """
        if self.positive_semidefinite:
            return self ** (0.5)
        else:
            raise ValueError("Non-positive semi-definite matrices cannot " +\
                "have their square roots taken because the result is not " +\
                "unique.")
    
    def inverse_square_root(self):
        """
        Finds \\(\\boldsymbol{M}^{-1/2}\\) by finding the inverse square roots
        of the individual blocks, i.e. finding \\(\\boldsymbol{A}_1^{-1/2},\
        \\boldsymbol{A}_2^{-1/2}, \\ldots, \\boldsymbol{A}_N^{-1/2}\\). This
        can only be done when
        `SparseSquareBlockDiagonalMatrix.positive_semidefinite` is True.
        
        Returns
        -------
        inverse_square_root_matrix : `SparseSquareBlockDiagonalMatrix`
            block diagonal matrix composed of inverse square root blocks:
            $$\\boldsymbol{M}^{-1/2}=\\begin{bmatrix}\
            \\boldsymbol{A}_1^{-1/2} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{0} \\\\ \\boldsymbol{0} & \\boldsymbol{A}_2^{-1/2} &\
            \\cdots & \\boldsymbol{0} \\\\ \\vdots & \\vdots & \\ddots &\
            \\vdots \\\\ \\boldsymbol{0} & \\boldsymbol{0} & \\cdots &\
            \\boldsymbol{A}_N^{-1/2} \\end{bmatrix}$$
        """
        if self.positive_semidefinite:
            return self ** (-0.5)
        else:
            raise ValueError("Non-positive semi-definite matrices cannot " +\
                "have their square roots taken because the result is not " +\
                "unique.")
    
    def inverse(self):
        """
        Finds \\(\\boldsymbol{M}^{-1}\\) by inverting the individual blocks,
        i.e. finding \\(\\boldsymbol{A}_1^{-1}, \\boldsymbol{A}_2^{-1},\
        \\ldots, \\boldsymbol{A}_N^{-1}\\).
        
        Returns
        -------
        inverse_matrix : `SparseSquareBlockDiagonalMatrix`
            block diagonal matrix composed of inverse blocks:
            $$\\boldsymbol{M}^{-1}=\\begin{bmatrix} \\boldsymbol{A}_1^{-1} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{0} \\\\ \\boldsymbol{0} &\
            \\boldsymbol{A}_2^{-1} & \\cdots & \\boldsymbol{0} \\\\ \\vdots &\
            \\vdots & \\ddots & \\vdots \\\\ \\boldsymbol{0} &\
            \\boldsymbol{0} & \\cdots & \\boldsymbol{A}_N^{-1} \\end{bmatrix}$$
        """
        if self.efficient:
            return SparseSquareBlockDiagonalMatrix(npla.inv(self.blocks))
        else:
            return SparseSquareBlockDiagonalMatrix(\
                [npla.inv(block) for block in self.blocks])
    
    def trace(self):
        """
        Finds the trace of the matrix by summing the diagonal elements of each
        block.
        
        Returns
        -------
        trace_value : float
            the trace of the matrix, \\(\\text{tr}(\\boldsymbol{M}) =\
            \\sum_{k=1}^N \\text{tr}(\\boldsymbol{A}_k)\\)
        """
        if self.efficient:
            trace_value = np.trace(self.blocks, axis1=1, axis2=2)
        else:
            trace_value = 0
            for block in self.blocks:
                trace_value = trace_value + np.trace(block)
        return trace_value
    
    def sign_and_log_abs_determinant(self):
        """
        Finds both the sign and logarithm of the absolute value of the
        determinant of this matrix. Essentially equivalent to
        `numpy.linalg.slogdet`.
        
        Returns
        -------
        sign : float
            the sign of the determinant, +1, -1, or 0. Since the determinant of
            a block diagonal matrix is the product of the determinants of the
            blocks, if \\(s_k\\) is the sign of the determinant of the
            \\(k^{\\text{th}}\\) block, \\(\\boldsymbol{A}_k\\), then the final
            returned sign, \\(s\\), is \\(s=\\prod_{k=1}^Ns_k\\)
        log_abs_determinant : float
            the natural logarithm of the absolute value of the determinant.
            Since the determinant of a block diagonal matrix is the product of
            the determinants of the blocks, `log_abs_determinant` is
            \\(\\ln{\\Vert\\boldsymbol{M}\\Vert} =\
            \\sum_{k=1}^N\\ln{\\Vert\\boldsymbol{A}_k\\Vert}\\)
        """
        if self.efficient:
            (signs, log_abs_determinants) = npla.slogdet(self.blocks)
            return (np.prod(signs), np.sum(log_abs_determinants))
        else:
            (sign, log_abs_determinant) = (1, 0)
            for block in self.blocks:
                (this_sign, this_log_abs_determinant) = npla.slogdet(block)
                sign = sign * this_sign
                log_abs_determinant =\
                    log_abs_determinant + this_log_abs_determinant
            return (sign, log_abs_determinant)
    
    def transpose(self):
        """
        Returns a transposed version of this matrix. If
        `SparseSquareBlockDiagonalMatrix.symmetric` is True, then a deep copy
        of this object is returned.
        """
        if self.symmetric:
            return self.copy()
        elif self.efficient:
            return SparseSquareBlockDiagonalMatrix(\
                np.swapaxes(self.blocks, -2, -1))
        else:
            return SparseSquareBlockDiagonalMatrix(\
                [block.T for block in self.blocks])
    
    @property
    def diagonal(self):
        """
        The diagonal elements of this matrix in a 1D `numpy.ndarray`.
        """
        if not hasattr(self, '_diagonal'):
            if self.efficient:
                self._diagonal =\
                    np.concatenate(np.diagonal(self.blocks, axis1=1, axis2=2))
            else:
                diagonal = []
                for block in self.blocks:
                    diagonal.append(np.diag(block))
                self._diagonal = np.concatenate(diagonal)
        return self._diagonal
    
    def copy(self):
        """
        Performs a deep copy of this `SparseSquareBlockDiagonalMatrix`.
        
        Returns
        -------
        copied : `SparseSquareBlockDiagonalMatrix`
            a deep copy of this object representing the same matrix
        """
        if self.efficient:
            return SparseSquareBlockDiagonalMatrix(self.blocks.copy())
        else:
            return SparseSquareBlockDiagonalMatrix(\
                [block.copy() for block in self.blocks])
    
    def __eq__(self, other):
        """
        Checks if `other` is a `SparseSquareBlockDiagonalMatrix` with the same
        matrix representation, even if blocked differently.
        
        Parameters
        ----------
        other : object
            other object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `SparseSquareBlockDiagonalMatrix`
            with the same matrx representation, even if blocked differently
        """
        if isinstance(other, SparseSquareBlockDiagonalMatrix):
            return SparseSquareBlockDiagonalMatrix.combine_blocks(self.blocks,\
                other.blocks, '==')
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with info about this
        `SparseSquareBlockDiagonalMatrix` so that it can be loaded later.
        
        Parameters
        ----------
        group: h5py.Group
            hdf5 file group to fill with information about this object
        """
        group.attrs['class'] = 'SparseSquareBlockDiagonalMatrix'
        if self.efficient:
            create_hdf5_dataset(group, 'blocks', data=self.blocks)
        else:
            for (iblock, block) in enumerate(self.blocks):
                create_hdf5_dataset(group, 'block{:d}'.format(iblock),\
                    data=block)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        A function which loads a `SparseSquareBlockDiagonalMatrix` from the
        given hdf5 file group.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group from which to load a
            `SparseSquareBlockDiagonalMatrix`
        
        Returns
        -------
        obj : `SparseSquareBlockDiagonalMatrix`
            a matrix loaded from the hdf5 file group
        """
        if 'blocks' in group:
            blocks = get_hdf5_value(group['blocks'])
        else:
            (index, blocks) = (0, [])
            while 'block{:d}'.format(index) in group:
                blocks.append(get_hdf5_value(group['block{:d}'.format(index)]))
                index += 1
        return SparseSquareBlockDiagonalMatrix(blocks)

