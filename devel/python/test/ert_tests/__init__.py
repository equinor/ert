#  Copyright (C) 2012  Statoil ASA, Norway.
#
#  The file '__init__.py' is part of ERT - Ensemble based Reservoir Tool.
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
#import os
#import traceback
#from ert_tests.source_enumerator import SourceEnumerator
#from ert.test import ExtendedTestCase
#
#
#class ErtTestCase(ExtendedTestCase):
#
#
#
#
#
#    def createTestPath(self, path):
#        """
#        @param: The test root path can be set by environment variable ERT_TEST_ROOT_PATH
#        """
#        file_path = os.path.realpath(__file__)
#        default_test_root = os.path.realpath(os.path.join(os.path.dirname(file_path), "../../../test-data/"))
#        test_root = os.path.realpath(os.environ.get("ERT_TEST_ROOT_PATH", default_test_root))
#        
#        return super(ErtTestCase , self).createTestPath(test_root , path)
#
