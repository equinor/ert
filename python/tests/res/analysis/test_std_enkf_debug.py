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
import sys

import res
from tests import ResTest
from res.analysis import AnalysisModule, AnalysisModuleLoadStatusEnum, AnalysisModuleOptionsEnum
from ecl.util.enums import RngAlgTypeEnum, RngInitModeEnum
from ecl.util.util.rng import RandomNumberGenerator


class StdEnKFDebugTest(ResTest):

    def setUp(self):
        self.rng = RandomNumberGenerator(RngAlgTypeEnum.MZRAN, RngInitModeEnum.INIT_DEFAULT)
        if sys.platform.lower() == 'darwin':
            self.libname = res.res_lib_path + "/std_enkf_debug.dylib"
        else:
            self.libname = res.res_lib_path + "/std_enkf_debug.so"
        self.module = AnalysisModule(lib_name = self.libname)


    def toggleKey(self, key):
        self.assertTrue( self.module.hasVar( key ))

        # check it is true
        self.assertTrue( self.module.setVar( key , True ) )
        self.assertTrue( self.module.getBool(key) )

        # set it to false
        self.assertTrue( self.module.setVar( key , False ) )
        self.assertFalse( self.module.getBool(key) )

    def test_EE_option(self):
        self.toggleKey( 'USE_EE' )


    def test_scaledata_option(self):
        self.toggleKey( 'ANALYSIS_SCALE_DATA' )

    def test_prefix(self):
        self.assertTrue( self.module.hasVar("PREFIX"))
        self.assertTrue( self.module.setVar( "PREFIX" , "Path") )
