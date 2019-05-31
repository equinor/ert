#!/usr/bin/env python
#  Copyright (C) 2014  Equinor ASA, Norway.
#
#  The file 'test_analysis_iter_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res.enkf import AnalysisIterConfig
from tests import ResTest


class AnalysisIterConfigTest(ResTest):

    def test_set_analysis_iter_config(self):
        c = AnalysisIterConfig()

        self.assertFalse( c.caseFormatSet() )
        c.setCaseFormat("case%d")
        self.assertTrue( c.caseFormatSet() )

        self.assertFalse( c.numIterationsSet() )
        c.setNumIterations(1)
        self.assertTrue( c.numIterationsSet() )

    def test_analysis_iter_config_constructor(self):
        config_dict = {
            'ITER_CASE':'ITERATED_ENSEMBLE_SMOOTHER%d',
            'ITER_COUNT':4,
            'ITER_RETRY_COUNT':4
        }
        c_default = AnalysisIterConfig()
        c_dict = AnalysisIterConfig(config_dict=config_dict)
        self.assertEqual(c_default, c_dict)



