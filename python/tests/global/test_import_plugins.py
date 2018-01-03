#  Copyright (C) 2017 Statoil ASA, Norway.
#
#  This file is part of ERT - Ensemble based Reservoir Tool.
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
import sys
import glob

from ecl.test import ImportTestCase

class ImportPlugins(ImportTestCase):

    def test_import(self):

        plugin_path = os.path.abspath( os.path.join( os.path.dirname( __file__) , "../../../../../../share/workflows/jobs/internal-gui/scripts") )
        print plugin_path
        self.assertTrue( os.path.isdir( plugin_path ))
        for path in glob.glob("%s/*.py" % plugin_path):
            print "Importing: %s" % path
            self.import_file( path )
