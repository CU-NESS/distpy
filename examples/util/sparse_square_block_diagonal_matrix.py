"""
Example showing and verifying features of the `SparseSquareBlockDiagonalMatrix`
class, including inverses, square roots, exponentiation, addition, subtraction,
negation, multiplication/division by scalars, and multiplication by matrices.

**File**: $DISTPY/examples/util/sparse_square_block_diagonal_matrix.py  
**Author**: Keith Tauscher  
**Date**: 21 May 2021
"""
from __future__ import division
import os
import numpy as np
import scipy.linalg as scila
from scipy import sparse
from distpy import SparseSquareBlockDiagonalMatrix

blocks = [np.array([[10, -6], [-6, 10]]), np.array([[1]])]
matrix1 = SparseSquareBlockDiagonalMatrix(blocks)
identity = SparseSquareBlockDiagonalMatrix(np.ones((3, 1, 1)))
assert(np.all(matrix1.dense() == scila.block_diag(*blocks)))
assert(matrix1.symmetric)
assert(matrix1.positive_semidefinite)
assert(not matrix1.efficient)
assert(matrix1 == matrix1.copy())
assert(matrix1 == matrix1.transpose())
assert(matrix1 == (matrix1 ** 1))
assert((matrix1 ** 0) == identity)
assert((-matrix1) ==\
    SparseSquareBlockDiagonalMatrix([(-block) for block in blocks]))
assert(np.allclose(matrix1.sign_and_log_abs_determinant(), (1, np.log(64))))
assert(np.isclose(matrix1.trace(), 21))
try:
    matrix1.save('THISISATEST.hdf5')
    assert(matrix1 == SparseSquareBlockDiagonalMatrix.load('THISISATEST.hdf5'))
except:
    os.remove('THISISATEST.hdf5')
    raise
else:
    os.remove('THISISATEST.hdf5')

expected_square_root_matrix1 =\
    SparseSquareBlockDiagonalMatrix([[[3, -1], [-1, 3]], [[1]]])
assert(matrix1.square_root() == expected_square_root_matrix1)

expected_inverse_matrix1 =\
    SparseSquareBlockDiagonalMatrix([np.array([[5, 3], [3, 5]]) / 32, [[1]]])
assert(matrix1.inverse() == expected_inverse_matrix1)

assert(matrix1.square_root().inverse() == matrix1.inverse_square_root())

expected_exponentiated_matrix1_base_2 = SparseSquareBlockDiagonalMatrix(\
    [[[32776, -32760], [-32760, 32776]], [[2]]])
assert((2 ** matrix1) == expected_exponentiated_matrix1_base_2)

matrix2_blocks = np.array([[[1]], [[0]], [[-1]]])
matrix2 = SparseSquareBlockDiagonalMatrix(matrix2_blocks)
assert(matrix2.symmetric)
assert(not matrix2.positive_semidefinite)
assert(matrix2.efficient)
assert((matrix2 * 2.5) ==\
    SparseSquareBlockDiagonalMatrix(2.5 * matrix2_blocks))
assert((matrix2 * 2.5) == (2.5 * matrix2))
assert((matrix2 / 2.) ==\
    SparseSquareBlockDiagonalMatrix(matrix2_blocks / 2))
try:
    matrix2 / 0.
except ZeroDivisionError:
    pass
else:
    raise AssertionError
try:
    matrix2.save('THISISATEST.hdf5')
    assert(matrix2 == SparseSquareBlockDiagonalMatrix.load('THISISATEST.hdf5'))
except:
    os.remove('THISISATEST.hdf5')
    raise
else:
    os.remove('THISISATEST.hdf5')

expected_exponentiated_matrix2 =\
    SparseSquareBlockDiagonalMatrix([[[np.e]], [[1]], [[1 / np.e]]])
assert(expected_exponentiated_matrix2 == (np.e ** matrix2))

try:
    matrix2.square_root()
except ValueError:
    pass
else:
    raise AssertionError

expected_12_matrix =\
    SparseSquareBlockDiagonalMatrix([[[10, 0], [-6, 0]], [[-1]]])
expected_21_matrix =\
    SparseSquareBlockDiagonalMatrix([[[10, -6], [0, 0]], [[-1]]])
assert(matrix1.__matmul__(matrix2) == expected_12_matrix)
assert(matrix2.__matmul__(matrix1) == expected_21_matrix)
assert(((matrix2.transpose().__matmul__(matrix1)).__matmul__(matrix2)) ==\
    (matrix2.transpose().__matmul__(matrix1.__matmul__(matrix2))))
assert((matrix2.transpose().__matmul__(matrix1.__matmul__(matrix2))).symmetric)

expected_1p2_matrix =\
    SparseSquareBlockDiagonalMatrix([[[11, -6], [-6, 10]], [[0]]])
assert((matrix1 + matrix2) == expected_1p2_matrix)

expected_1m2_matrix =\
    SparseSquareBlockDiagonalMatrix([[[9, -6], [-6, 10]], [[2]]])
assert((matrix1 - matrix2) == expected_1m2_matrix)

assert(np.all((matrix2.__matmul__(np.identity(3))) == matrix2.dense()))

random_matrix = np.random.normal(0, 1, size=(3,3))
assert(np.all((matrix2.__matmul__(random_matrix)) ==\
    np.matmul(matrix2.dense(), random_matrix.T).T))
assert(np.all((matrix2.__matmul__(random_matrix)) ==\
    matrix2.array_matrix_multiplication(random_matrix)))
assert(np.all(matrix2.array_matrix_multiplication(np.identity(3),\
    right=False) == matrix2.dense()))

