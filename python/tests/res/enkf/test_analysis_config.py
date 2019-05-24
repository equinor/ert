#!/usr/bin/env python
#  Copyright (C) 2013  Equinor ASA, Norway.
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

from ecl.util.test import TestAreaContext
from tests import ResTest

from res.enkf import AnalysisConfig
from res.enkf import ConfigKeys

class AnalysisConfigTest(ResTest):

    def setUp(self):
        self.case_directory = self.createTestPath("local/simple_config/")
        self.case_file = 'simple_config/minimum_config'

    def test_invalid_user_config(self):
        with TestAreaContext("void land"):
            with self.assertRaises(IOError):
                AnalysisConfig("this/is/not/a/file")

    def test_keywords_for_monitoring_simulation_runtime(self):
        with TestAreaContext("analysis_config_init_test") as work_area:
            work_area.copy_directory(self.case_directory)
            ac = AnalysisConfig(self.case_file)

            # Unless the MIN_REALIZATIONS is set in config, one is required to have "all" realizations.
            self.assertFalse(ac.haveEnoughRealisations(5, 10))
            self.assertTrue(ac.haveEnoughRealisations(10, 10))

            ac.set_max_runtime( 50 )
            self.assertEqual( 50 , ac.get_max_runtime() )

            ac.set_stop_long_running( True )
            self.assertTrue( ac.get_stop_long_running() )


    def test_analysis_modules(self):
        with TestAreaContext("analysis_config_init_test") as work_area:
            work_area.copy_directory(self.case_directory)
            ac = AnalysisConfig(self.case_file)
            self.assertIsNotNone( ac.activeModuleName() )
            self.assertIsNotNone( ac.getModuleList() )

    def test_analysis_config_global_std_scaling(self):
        with TestAreaContext("analysis_config_init_test") as work_area:
            work_area.copy_directory(self.case_directory)
            ac = AnalysisConfig(self.case_file)
            self.assertFloatEqual(ac.getGlobalStdScaling(), 1.0)
            ac.setGlobalStdScaling(0.77)
            self.assertFloatEqual(ac.getGlobalStdScaling(), 0.77)

    def test_init(self):
        with TestAreaContext("analysis_config_init_test") as work_area:
            work_area.copy_directory(self.case_directory)
            analysis_config = AnalysisConfig(self.case_file)
            self.assertIsNotNone(analysis_config)

    def test_analysis_config_constructor(self):
        with TestAreaContext("analysis_config_constructor_test") as work_area:
            work_area.copy_directory(self.case_directory)
            config_dict = {
                ConfigKeys.ALPHA_KEY: 3,
                ConfigKeys.RERUN_KEY: False,
                ConfigKeys.RERUN_START_KEY: 0,
                ConfigKeys.MERGE_OBSERVATIONS: False,
                ConfigKeys.UPDATE_LOG_PATH: 'update_log',
                ConfigKeys.STD_CUTOFF_KEY: 1e-6,
                ConfigKeys.STOP_LONG_RUNNING: False,
                ConfigKeys.SINGLE_NODE_UPDATE: False,
                ConfigKeys.STD_CORRELATED_OBS: False,
                ConfigKeys.GLOBAL_STD_SCALING: 1,
                ConfigKeys.MAX_RUNTIME: 0,
                ConfigKeys.MIN_REALIZATIONS: 0,
                ConfigKeys.ANALYSIS_LOAD: [
                    {
                        ConfigKeys.USER_NAME: 'RML_ENKF',
                        ConfigKeys.LIB_NAME: 'rml_enkf.so'
                    },
                    {
                        ConfigKeys.USER_NAME: 'MODULE_ENKF',
                        ConfigKeys.LIB_NAME: 'rml_enkf.so'
                    }
                ],
                ConfigKeys.ANALYSIS_COPY: [
                    {
                        ConfigKeys.SRC_NAME:'STD_ENKF',
                        ConfigKeys.DST_NAME:'ENKF_HIGH_TRUNCATION'
                    }
                ],
                ConfigKeys.ANALYSIS_SET_VAR:[
                    {
                        ConfigKeys.MODULE_NAME:'STD_ENKF',
                        ConfigKeys.VAR_NAME:'ENKF_NCOMP',
                        ConfigKeys.VALUE:2
                    },
                    {
                        ConfigKeys.MODULE_NAME: 'ENKF_HIGH_TRUNCATION',
                        ConfigKeys.VAR_NAME: 'ENKF_TRUNCATION',
                        ConfigKeys.VALUE: 0.99
                    }
                ],
                ConfigKeys.ANALYSIS_SELECT:'ENKF_HIGH_TRUNCATION'
            }
            _config_file = 'simple_config/analysis_config'
            analysis_config_file = AnalysisConfig(user_config_file=_config_file)
            analysis_config_dict = AnalysisConfig(config_dict=config_dict)
            self.assertEqual(analysis_config_dict, analysis_config_file)


