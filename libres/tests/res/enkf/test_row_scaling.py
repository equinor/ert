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
import random
from functools import partial
from res.enkf import RowScaling

import math
from ecl.grid import EclGridGenerator
from res.enkf import FieldConfig


def row_scaling_one(data_index):
    return 1.0


def row_scaling_inverse_size(size, data_index):
    return data_index / size


def row_scaling_coloumb(nx, ny, data_index):
    j = data_index / nx
    i = data_index - j * nx

    dx = 0.5 + i
    dy = 0.5 + j

    r2 = dx * dx + dy * dy
    return 0.50 / r2


def gaussian_decay(obs_pos, length_scale, grid, data_index):
    x, y, z = grid.get_xyz(active_index=data_index)
    dx = (obs_pos[0] - x) / length_scale[0]
    dy = (obs_pos[1] - y) / length_scale[1]
    dz = (obs_pos[2] - z) / length_scale[2]

    exp_arg = -0.5 * (dx * dx + dy * dy + dz * dz)
    return math.exp(exp_arg)


def test_basic():
    row_scaling = RowScaling()
    assert len(row_scaling) == 0

    row_scaling[9] = 0.25
    assert len(row_scaling) == 10
    assert row_scaling[0] == 0
    assert row_scaling[9] == 0.25

    with pytest.raises(IndexError):
        var = row_scaling[10]

    for i in range(len(row_scaling)):
        r = random.random()
        row_scaling[i] = r
        assert row_scaling[i] == row_scaling.clamp(r)

    nx = 10
    ny = 10
    row_scaling.assign(nx * ny, row_scaling_one)
    assert len(row_scaling) == nx * ny
    assert row_scaling[0] == 1
    assert row_scaling[nx * ny - 1] == 1

    inverse_size = partial(row_scaling_inverse_size, nx * ny)
    row_scaling.assign(nx * ny, inverse_size)
    for g in range(nx * ny):
        assert row_scaling[g] == row_scaling.clamp(g / (nx * ny))

    coloumb = partial(row_scaling_coloumb, nx, ny)
    row_scaling.assign(nx * ny, coloumb)
    for j in range(ny):
        for i in range(nx):
            g = j * nx + i
            assert row_scaling[g] == row_scaling.clamp(row_scaling_coloumb(nx, ny, g))

    with pytest.raises(TypeError):
        row_scaling.assign_vector(123.0)


def test_field_config():
    nx = 10
    ny = 10
    nz = 5
    actnum = [1] * nx * ny * nz
    actnum[0] = 0
    actnum[3] = 0
    actnum[10] = 0

    grid = EclGridGenerator.create_rectangular((nx, ny, nz), (1, 1, 1), actnum)
    fc = FieldConfig("PORO", grid)
    row_scaling = RowScaling()
    obs_pos = grid.get_xyz(ijk=(5, 5, 1))
    length_scale = (2, 1, 0.50)

    gaussian = partial(gaussian_decay, obs_pos, length_scale, grid)
    row_scaling.assign(grid.get_num_active(), gaussian)
    for g in range(grid.get_num_active()):
        assert row_scaling[g] == row_scaling.clamp(
            gaussian_decay(obs_pos, length_scale, grid, g)
        )
