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

from libres_utils import ResTest

from res.analysis import AnalysisModuleOptionsEnum


class AnalysisOptionsEnumTest(ResTest):
    def test_items(self):

        assert AnalysisModuleOptionsEnum.ANALYSIS_USE_A == 4
        assert AnalysisModuleOptionsEnum.ANALYSIS_UPDATE_A == 8
        assert AnalysisModuleOptionsEnum.ANALYSIS_ITERABLE == 32
