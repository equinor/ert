#!/usr/bin/env python
#  Copyright (C) 2013  Statoil ASA, Norway.
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

from ert_tests import ExtendedTestCase
from ert.analysis import AnalysisModuleOptionsEnum


class AnalysisOptionsEnumTest(ExtendedTestCase):
    def test_items(self):
        self.assertTrue( AnalysisModuleOptionsEnum.has_element( "ANALYSIS_NEED_ED" ))
        self.assertTrue( AnalysisModuleOptionsEnum.has_element( "ANALYSIS_USE_A" ))
        self.assertTrue( AnalysisModuleOptionsEnum.has_element( "ANALYSIS_UPDATE_A" ))
        self.assertTrue( AnalysisModuleOptionsEnum.has_element( "ANALYSIS_SCALE_DATA" ))
        self.assertTrue( AnalysisModuleOptionsEnum.has_element( "ANALYSIS_ITERABLE" ))
