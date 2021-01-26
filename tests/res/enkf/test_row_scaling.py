#  Copyright (C) 2020  Equinor ASA, Norway.
#
#  The file 'test_row_scaling.py' is part of ERT - Ensemble based Reservoir Tool.
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

import pytest
from res.enkf import RowScaling


def test_access():
    row_scaling = RowScaling()
    assert len(row_scaling) == 0

    row_scaling[9] = 0.25
    assert len(row_scaling) == 10
    assert row_scaling[0] == 1
    assert row_scaling[9] == 0.25

    with pytest.raises(IndexError):
        var = row_scaling[10]
