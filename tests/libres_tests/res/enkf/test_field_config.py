# Copyright (C) 2017  Equinor ASA, Norway.
#
# This file is part of ERT - Ensemble based Reservoir Tool.
#
# ERT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ERT is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
# for more details.

import pytest
from ecl.grid import EclGridGenerator

from ert._c_wrappers.enkf.config import FieldConfig, FieldTypeEnum
from ert._c_wrappers.enkf.enums import EnkfFieldFileFormatEnum


def test_create():
    FieldConfig("SWAT", EclGridGenerator.create_rectangular((10, 10, 5), (1, 1, 1)))


def test_field_guess_filetype(tmp_path):
    fname = str(tmp_path / "test.kw.grdecl")
    with open(fname, "w") as f:
        f.write("-- my comment\n")
        f.write("-- more comments\n")
        f.write("SOWCR\n")
        # The function guessing file types determines whether the file
        # is binary or 7 bit ASCII based on bit 8 heuristics. For this
        # to be "reliable" the file is required to be more than 256
        # bytes.
        for _ in range(256 // 8):
            f.write("0 0 0 0\n")

    ft = FieldConfig.guessFiletype(fname)
    grdecl_type = EnkfFieldFileFormatEnum(5)
    # pylint: disable=no-member
    assert grdecl_type.name == "ECL_GRDECL_FILE"
    assert grdecl_type == ft


def test_field_type_enum():
    assert FieldTypeEnum.ECLIPSE_PARAMETER == FieldTypeEnum(2)
    gen = FieldTypeEnum.GENERAL
    assert str(gen) == "GENERAL"
    gen = FieldTypeEnum(3)
    assert str(gen) == "GENERAL"


def test_export_format():
    assert (
        FieldConfig.exportFormat("file.grdecl")
        == EnkfFieldFileFormatEnum.ECL_GRDECL_FILE
    )
    assert (
        FieldConfig.exportFormat("file.xyz.grdecl")
        == EnkfFieldFileFormatEnum.ECL_GRDECL_FILE
    )
    assert (
        FieldConfig.exportFormat("file.roFF") == EnkfFieldFileFormatEnum.RMS_ROFF_FILE
    )
    assert (
        FieldConfig.exportFormat("file.xyz.roFF")
        == EnkfFieldFileFormatEnum.RMS_ROFF_FILE
    )

    with pytest.raises(ValueError):
        FieldConfig.exportFormat("file.xyz")

    with pytest.raises(ValueError):
        FieldConfig.exportFormat("file.xyz")


def test_basics():
    nx = 17
    ny = 13
    nz = 11
    actnum = [1] * nx * ny * nz
    actnum[0] = 0
    grid = EclGridGenerator.create_rectangular((nx, ny, nz), (1, 1, 1), actnum)
    fc = FieldConfig("PORO", grid)
    assert repr(fc).startswith("FieldConfig(type")
    fc_xyz = fc.get_nx(), fc.get_ny(), fc.get_nz()
    ex_xyz = nx, ny, nz
    assert ex_xyz == fc_xyz
    assert fc.get_truncation_mode() == 0
    assert ex_xyz == (grid.getNX(), grid.getNY(), grid.getNZ())
    assert fc.get_data_size() == grid.get_num_active()
