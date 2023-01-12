import pytest
from ecl.grid import EclGridGenerator

from ert._c_wrappers.enkf.config import FieldConfig, FieldTypeEnum
from ert._c_wrappers.enkf.enums import EnkfFieldFileFormatEnum


def test_create():
    FieldConfig("SWAT", EclGridGenerator.create_rectangular((10, 10, 5), (1, 1, 1)))


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
