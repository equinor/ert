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

import os

import pytest
from hypothesis import given
from libres_utils import ResTest, tmpdir
from res.enkf import ConfigKeys, SiteConfig

from config_dict_generator import config_dicts, to_config_file


def test_site_nonexistent_file():
    with pytest.raises(IOError):
        SiteConfig("does/not/exist")


@pytest.mark.usefixtures("setup_tmpdir")
@given(config_dicts())
def test_site_config_dict_same_as_from_file(config_dict):
    cwd = os.getcwd()
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    assert SiteConfig(filename) == SiteConfig(config_dict=config_dict)


class SiteConfigTest(ResTest):
    @tmpdir()
    def test_site_config_hook_workflow(self):
        site_config_filename = "test_site_config"
        test_config_filename = "test_config"
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

        with open("ECHO_WORKFLOW_JOB", "w") as fh:
            fh.write(
                """INTERNAL False
EXECUTABLE echo
MIN_ARG 1
"""
            )

        with open("ECHO_WORKFLOW", "w") as fh:
            fh.write("ECHO_WORKFLOW_JOB hello")

        self.monkeypatch.setenv("ERT_SITE_CONFIG", site_config_filename)

        res_config = ResConfig(user_config_file=test_config_filename)
        self.assertTrue(len(res_config.hook_manager) == 1)
        self.assertEqual(
            res_config.hook_manager[0].getWorkflow().src_file,
            os.path.join(os.getcwd(), "ECHO_WORKFLOW"),
        )
