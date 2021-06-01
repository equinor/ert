#!/usr/bin/env python
#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  The file 'test_options_enum.py' is part of ERT - Ensemble based Reservoir Tool.
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

from tests import ResTest
from res.analysis import AnalysisModuleOptionsEnum


class AnalysisOptionsEnumTest(ResTest):
    def test_items(self):
        source_file_path = "lib/include/ert/analysis/analysis_module.hpp"
        self.assertEnumIsFullyDefined(
            AnalysisModuleOptionsEnum, "analysis_module_flag_enum", source_file_path
        )
