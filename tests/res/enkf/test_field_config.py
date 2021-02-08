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
from os.path import abspath
from ecl.util.test import TestAreaContext
from tests import ResTest

from ecl.grid import EclGridGenerator
from res.enkf.config import FieldTypeEnum, FieldConfig
from res.enkf.enums import EnkfFieldFileFormatEnum


class FieldConfigTest(ResTest):
    def test_field_guess_filetype(self):
        with TestAreaContext("field_config") as test_context:
            fname = abspath("test.kw.grdecl")
            with open(fname, "w") as f:
                f.write("-- my comment\n")
                f.write("-- more comments\n")
                f.write("SOWCR\n")
                for i in range(128 // 8):
                    f.write("0 0 0 0\n")
                f.write("/\n")

            ft = FieldConfig.guessFiletype(fname)
            grdecl_type = EnkfFieldFileFormatEnum(5)
            self.assertEqual("ECL_GRDECL_FILE", grdecl_type.name)
            self.assertEqual(grdecl_type, ft)

    def test_field_type_enum(self):
        self.assertEqual(FieldTypeEnum(2), FieldTypeEnum.ECLIPSE_PARAMETER)
        gen = FieldTypeEnum.GENERAL
        self.assertEqual("GENERAL", str(gen))
        gen = FieldTypeEnum(3)
        self.assertEqual("GENERAL", str(gen))

    def test_export_format(self):
        self.assertEqual(
            FieldConfig.exportFormat("file.grdecl"),
            EnkfFieldFileFormatEnum.ECL_GRDECL_FILE,
        )
        self.assertEqual(
            FieldConfig.exportFormat("file.xyz.grdecl"),
            EnkfFieldFileFormatEnum.ECL_GRDECL_FILE,
        )
        self.assertEqual(
            FieldConfig.exportFormat("file.roFF"), EnkfFieldFileFormatEnum.RMS_ROFF_FILE
        )
        self.assertEqual(
            FieldConfig.exportFormat("file.xyz.roFF"),
            EnkfFieldFileFormatEnum.RMS_ROFF_FILE,
        )

        with self.assertRaises(ValueError):
            FieldConfig.exportFormat("file.xyz")

        with self.assertRaises(ValueError):
            FieldConfig.exportFormat("file.xyz")

    def test_basics(self):
        nx = 17
        ny = 13
        nz = 11
        actnum = [1] * nx * ny * nz
        actnum[0] = 0
        grid = EclGridGenerator.create_rectangular((nx, ny, nz), (1, 1, 1), actnum)
        fc = FieldConfig("PORO", grid)
        pfx = "FieldConfig(type"
        rep = repr(fc)
        self.assertEqual(pfx, rep[: len(pfx)])
        fc_xyz = fc.get_nx(), fc.get_ny(), fc.get_nz()
        ex_xyz = nx, ny, nz
        self.assertEqual(ex_xyz, fc_xyz)
        self.assertEqual(0, fc.get_truncation_mode())
        self.assertEqual(ex_xyz, (grid.getNX(), grid.getNY(), grid.getNZ()))
        self.assertEqual(fc.get_data_size(), grid.get_num_active())
