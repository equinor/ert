#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_site_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

import json
import os
from pathlib import Path

import pytest

from ert._c_wrappers.enkf import ConfigKeys, EnKFMain, ResConfig, SiteConfig

def test_constructors(snake_oil_case):
    ert_site_config = SiteConfig.getLocation()
    ert_share_path = os.path.dirname(ert_site_config)

    assert snake_oil_case.resConfig().site_config == SiteConfig(
        config_dict={
            ConfigKeys.INSTALL_JOB: [
                {
                    ConfigKeys.NAME: "SNAKE_OIL_SIMULATOR",
                    ConfigKeys.PATH: os.getcwd() + "/jobs/SNAKE_OIL_SIMULATOR",
                },
                {
                    ConfigKeys.NAME: "SNAKE_OIL_NPV",
                    ConfigKeys.PATH: os.getcwd() + "/jobs/SNAKE_OIL_NPV",
                },
                {
                    ConfigKeys.NAME: "SNAKE_OIL_DIFF",
                    ConfigKeys.PATH: os.getcwd() + "/jobs/SNAKE_OIL_DIFF",
                },
            ],
            ConfigKeys.INSTALL_JOB_DIRECTORY: [
                ert_share_path + "/forward-models/res",
                ert_share_path + "/forward-models/shell",
                ert_share_path + "/forward-models/templating",
                ert_share_path + "/forward-models/old_style",
            ],
            ConfigKeys.SETENV: [
                {ConfigKeys.NAME: "SILLY_VAR", ConfigKeys.VALUE: "silly-valye"},
                {
                    ConfigKeys.NAME: "OPTIONAL_VAR",
                    ConfigKeys.VALUE: "optional-value",
                },
            ],
            ConfigKeys.LICENSE_PATH: "some/random/path",
            ConfigKeys.UMASK: 18,
        }
    )


def test_2_inits_raises():
    with pytest.raises(ValueError):
        SiteConfig(config_content="a_config", config_dict={"some": "config"})


def test_site_config_hook_workflow(monkeypatch, tmp_path):
    site_config_filename = str(tmp_path / "test_site_config")
    test_config_filename = str(tmp_path / "test_config")
    site_config_content = """
QUEUE_SYSTEM LOCAL
LOAD_WORKFLOW_JOB ECHO_WORKFLOW_JOB
LOAD_WORKFLOW ECHO_WORKFLOW
HOOK_WORKFLOW ECHO_WORKFLOW PRE_SIMULATION
"""

    with open(site_config_filename, "w") as fh:
        fh.write(site_config_content)

    with open(test_config_filename, "w") as fh:
        fh.write("NUM_REALIZATIONS 1\n")

    with open(tmp_path / "ECHO_WORKFLOW_JOB", "w") as fh:
        fh.write(
            """INTERNAL False
EXECUTABLE echo
MIN_ARG 1
"""
        )

    with open(tmp_path / "ECHO_WORKFLOW", "w") as fh:
        fh.write("ECHO_WORKFLOW_JOB hello")

    monkeypatch.setenv("ERT_SITE_CONFIG", site_config_filename)

    res_config = ResConfig(user_config_file=test_config_filename)
    assert len(res_config.hook_manager) == 1
    assert res_config.hook_manager[0].getWorkflow().src_file == str(
        tmp_path / "ECHO_WORKFLOW"
    )
