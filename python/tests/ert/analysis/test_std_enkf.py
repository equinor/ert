#  Copyright (C) 2016  Statoil ASA, Norway.
#
#  The file 'test_analysis_module.py' is part of ERT - Ensemble based Reservoir Tool.
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

from ert.test import ExtendedTestCase
from ert.analysis import AnalysisModule, AnalysisModuleLoadStatusEnum, AnalysisModuleOptionsEnum

from ert.util.enums import RngAlgTypeEnum, RngInitModeEnum
from ert.util.rng import RandomNumberGenerator
import ert.cwrap.clib as clib


class StdEnKFTest(ExtendedTestCase):

    def setUp(self):
        self.rng = RandomNumberGenerator(RngAlgTypeEnum.MZRAN, RngInitModeEnum.INIT_DEFAULT)
        self.module = AnalysisModule( self.rng , name = "STD_ENKF" )


    def test_EE_option(self):
        self.assertTrue( self.module.setVar( "USE_EE" , True ) )
        
        
