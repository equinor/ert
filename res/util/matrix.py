#  Copyright (C) 2011  Equinor ASA, Norway.
#
#  The file 'matrix.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.


# The Matrix class implemented here wraps the C matrix implementation
# in matrix.c from the libutil library. The C matrix implementation
# has the very limited ambition of just barely satisfying the matrix
# needs of the EnKF algorithm, i.e. for general linear algebra
# applications you will probably be better served by a more complete
# matrix library. This applies even more so to this Python
# implementation; it is only here facilitate use of C libraries which
# expect a matrix instance as input (i.e. the LARS estimator). For
# general linear algebra in Python the numpy library is a natural
# choice.


from cwrap import BaseCClass
from res import ResPrototype


class Matrix(BaseCClass):
    _matrix_alloc = ResPrototype("void*  matrix_alloc(int, int )", bind=False)
    _matrix_alloc_identity = ResPrototype(
        "matrix_obj  matrix_alloc_identity( int )", bind=False
    )
    _alloc_transpose = ResPrototype("matrix_obj  matrix_alloc_transpose(matrix)")
    _inplace_transpose = ResPrototype("void        matrix_inplace_transpose(matrix)")
    _copy = ResPrototype("matrix_obj  matrix_alloc_copy(matrix)")
    _free = ResPrototype("void   matrix_free(matrix)")
    _iget = ResPrototype("double matrix_iget( matrix , int , int )")
    _iset = ResPrototype("void   matrix_iset( matrix , int , int , double)")
    _set_all = ResPrototype("void   matrix_set_all( matrix , double)")
    _scale_column = ResPrototype("void matrix_scale_column(matrix , int , double)")
    _scale_row = ResPrototype("void matrix_scale_row(matrix , int , double)")
    _copy_column = ResPrototype(
        "void matrix_copy_column(matrix , matrix , int , int)", bind=False
    )
    _rows = ResPrototype("int matrix_get_rows(matrix)")
    _columns = ResPrototype("int matrix_get_columns(matrix)")
    _equal = ResPrototype("bool matrix_equal(matrix, matrix)")
    _random_init = ResPrototype("void matrix_random_init(matrix, rng)")

    _alloc_matmul = ResPrototype(
        "matrix_obj  matrix_alloc_matmul(matrix, matrix)", bind=False
    )

    @classmethod
    def matmul(cls, m1, m2):
        """
        Will return a new matrix which is matrix product of m1 and m2.
        """
        if m1.columns() == m2.rows():
            return cls._alloc_matmul(m1, m2)
        else:
            raise ValueError("Matrix size mismatch")

    def __init__(self, rows, columns, value=0):
        c_ptr = self._matrix_alloc(rows, columns)
        super().__init__(c_ptr)
        self.setAll(value)

    def setAll(self, value):
        self._set_all(value)

    def copy(self):
        return self._copy()

    @classmethod
    def identity(cls, dim):
        """Returns a dim x dim identity matrix."""
        if dim < 1:
            raise ValueError(
                "Identity matrix must have positive size, %d not allowed." % dim
            )
        return cls._matrix_alloc_identity(dim)

    def __str__(self):
        s = ""
        for i in range(self.rows()):
            s += "["
            for j in range(self.columns()):
                d = self._iget(i, j)
                s += "%6.3g " % d
            s += "]\n"
        return s

    def __getitem__(self, index_tuple):
        if not 0 <= index_tuple[0] < self.rows():
            raise IndexError("Expected 0 <= %d < %d" % (index_tuple[0], self.rows()))

        if not 0 <= index_tuple[1] < self.columns():
            raise IndexError("Expected 0 <= %d < %d" % (index_tuple[1], self.columns()))

        return self._iget(index_tuple[0], index_tuple[1])

    def __setitem__(self, index_tuple, value):
        if not 0 <= index_tuple[0] < self.rows():
            raise IndexError("Expected 0 <= %d < %d" % (index_tuple[0], self.rows()))

        if not 0 <= index_tuple[1] < self.columns():
            raise IndexError("Expected 0 <= %d < %d" % (index_tuple[1], self.columns()))

        return self._iset(index_tuple[0], index_tuple[1], value)

    def dims(self):
        return self._rows(), self._columns()

    def rows(self):
        """@rtype: int"""
        return self._rows()

    def transpose(self, inplace=False):
        """
        Will transpose the matrix. By default a transposed copy is returned.
        """
        if inplace:
            self._inplace_transpose()
            return self
        else:
            return self._alloc_transpose()

    def columns(self):
        """@rtype: int"""
        return self._columns()

    def __eq__(self, other):
        assert isinstance(other, Matrix)
        return self._equal(other)

    def copyColumn(self, target_column, src_column):
        columns = self.columns()
        if not 0 <= src_column < columns:
            raise ValueError("src column:%d invalid" % src_column)

        if not 0 <= target_column < columns:
            raise ValueError("target column:%d invalid" % target_column)

        if src_column != target_column:
            # The underlying C function accepts column copy between matrices.
            Matrix._copy_column(self, self, target_column, src_column)

    def randomInit(self, rng):
        self._random_init(rng)

    def free(self):
        self._free()
