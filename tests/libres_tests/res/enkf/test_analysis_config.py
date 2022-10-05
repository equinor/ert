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

import pytest

from ert._c_wrappers.enkf import AnalysisConfig, ConfigKeys


@pytest.fixture
def analysis_config(minimum_case):
    return minimum_case.resConfig().analysis_config


def test_keywords_for_monitoring_simulation_runtime(analysis_config):

    # Unless the MIN_REALIZATIONS is set in config, one is required to
    # have "all" realizations.
    assert not analysis_config.haveEnoughRealisations(5)
    assert analysis_config.haveEnoughRealisations(10)

    analysis_config.set_max_runtime(50)
    assert analysis_config.get_max_runtime() == 50

    analysis_config.set_stop_long_running(True)
    assert analysis_config.get_stop_long_running()


def test_analysis_modules(analysis_config):
    assert analysis_config.activeModuleName() is not None
    assert analysis_config.getModuleList() is not None


def test_analysis_config_global_std_scaling(analysis_config):
    assert pytest.approx(analysis_config.getGlobalStdScaling()) == 1.0
    analysis_config.setGlobalStdScaling(0.77)
    assert pytest.approx(analysis_config.getGlobalStdScaling()) == 0.77


def test_analysis_config_constructor(setup_case):
    res_config = setup_case("simple_config", "analysis_config")
    assert res_config.analysis_config == AnalysisConfig(
        config_dict={
            ConfigKeys.NUM_REALIZATIONS: 10,
            ConfigKeys.ALPHA_KEY: 3,
            ConfigKeys.RERUN_KEY: False,
            ConfigKeys.RERUN_START_KEY: 0,
            ConfigKeys.UPDATE_LOG_PATH: "update_log",
            ConfigKeys.STD_CUTOFF_KEY: 1e-6,
            ConfigKeys.STOP_LONG_RUNNING: False,
            ConfigKeys.SINGLE_NODE_UPDATE: False,
            ConfigKeys.GLOBAL_STD_SCALING: 1,
            ConfigKeys.MAX_RUNTIME: 0,
            ConfigKeys.MIN_REALIZATIONS: 0,
            ConfigKeys.ANALYSIS_COPY: [
                {
                    ConfigKeys.SRC_NAME: "STD_ENKF",
                    ConfigKeys.DST_NAME: "ENKF_HIGH_TRUNCATION",
                }
            ],
            ConfigKeys.ANALYSIS_SET_VAR: [
                {
                    ConfigKeys.MODULE_NAME: "STD_ENKF",
                    ConfigKeys.VAR_NAME: "ENKF_NCOMP",
                    ConfigKeys.VALUE: 2,
                },
                {
                    ConfigKeys.MODULE_NAME: "ENKF_HIGH_TRUNCATION",
                    ConfigKeys.VAR_NAME: "ENKF_TRUNCATION",
                    ConfigKeys.VALUE: 0.99,
                },
            ],
            ConfigKeys.ANALYSIS_SELECT: "ENKF_HIGH_TRUNCATION",
        }
    )
