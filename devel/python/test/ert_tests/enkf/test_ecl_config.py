#!/usr/bin/env python
#  Copyright (C) 2013  Statoil ASA, Norway.
#
#  The file 'test_analysis_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

import os
import os.path
from ert.enkf import EclConfig
from ert_tests import ExtendedTestCase
from ert.util import UIReturn

EGRID_file  = "Statoil/ECLIPSE/Gurbat/ECLIPSE.EGRID"
SMSPEC_file = "Statoil/ECLIPSE/Gurbat/ECLIPSE.SMSPEC"

class EclConfigTest(ExtendedTestCase):


    def test_grid(self):
        grid_file = self.createTestPath( EGRID_file )
        smspec_file = self.createTestPath( SMSPEC_file )
        ec = EclConfig()
        ui = ec.validateGridFile( grid_file )
        self.assertTrue( ui )
        self.assertTrue( isinstance(ui , UIReturn ))

        ui = ec.validateGridFile( "Does/Not/Exist" )
        self.assertFalse( ui )
        
        self.assertTrue( os.path.exists( smspec_file )) 
        ui = ec.validateGridFile( smspec_file )
        self.assertFalse( ui )



    def test_eclbase(self):
        ec = EclConfig()
        ui = ec.validateEclBase( "MixedCase%d" )
        self.assertFalse( ui )

        ui = ec.validateEclBase( "CASE%s" )
        self.assertFalse( ui )

        ui = ec.validateEclBase( "CASE%d" )
        self.assertTrue( ui )
        ec.setEclBase("CASE%d")
        self.assertEqual( "CASE%d" , ec.getEclBase())
        
    
