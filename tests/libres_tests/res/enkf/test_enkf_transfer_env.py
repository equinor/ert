#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_enkf_transfer_env.py' is part of ERT - Ensemble based Reservoir Tool.
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

import pytest
from ecl.util.test import TestAreaContext
from ecl.util.util import BoolVector
from libres_utils import ResTest, tmpdir

from res.enkf.ert_run_context import ErtRunContext
from res.test import ErtTestContext


@pytest.mark.unstable
class EnKFTestTransferEnv(ResTest):
    def setUp(self):
        pass

    @tmpdir()
    def test_transfer_var(self):

        with TestAreaContext("enkf_test_transfer_env") as work_area:
            base_path = os.getcwd()
            source_path = self.createTestPath("local/snake_oil_no_data")

            work_area.copy_directory(source_path)
            dir_ert = os.path.join(base_path, "snake_oil_no_data")
            assert os.path.isdir(dir_ert)

            file_ert = os.path.join(dir_ert, "snake_oil.ert")
            assert os.path.isfile(file_ert)

            with ErtTestContext("transfer_env_var", model_config=file_ert) as ctx:
                ert = ctx.getErt()
                fs_manager = ert.getEnkfFsManager()
                result_fs = fs_manager.getCurrentFileSystem()

                model_config = ert.getModelConfig()
                runpath_fmt = model_config.getRunpathFormat()
                jobname_fmt = model_config.getJobnameFormat()
                subst_list = ert.getDataKW()
                itr = 0
                mask = BoolVector(default_value=True, initial_size=1)
                run_context = ErtRunContext.ensemble_experiment(
                    result_fs, mask, runpath_fmt, jobname_fmt, subst_list, itr
                )
                ert.getEnkfSimulationRunner().createRunPath(run_context)
                os.chdir("storage/snake_oil/runpath/realization-0/iter-0")
                assert os.path.isfile("jobs.json")
                with open("jobs.json", "r") as f:
                    data = json.load(f)
                    env_data = data["global_environment"]
                    self.assertEqual("TheFirstValue", env_data["FIRST"])
                    self.assertEqual("TheSecondValue", env_data["SECOND"])

                    path_data = data["global_update_path"]
                    self.assertEqual("TheThirdValue", path_data["THIRD"])
                    self.assertEqual("TheFourthValue", path_data["FOURTH"])
