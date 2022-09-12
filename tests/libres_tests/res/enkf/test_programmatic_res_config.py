#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_programmatic_res_config.py' is part of ERT.
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
from textwrap import dedent

import pytest

from ert._c_wrappers.enkf import ResConfig


def test_new_config(tmp_path):
    os.chdir(tmp_path)
    with open(tmp_path / "NEW_TYPE_A", "w") as fout:
        fout.write(
            dedent(
                """
        EXECUTABLE echo
        MIN_ARG    1
        MAX_ARG    4
        ARG_TYPE 0 STRING
        ARG_TYPE 1 BOOL
        ARG_TYPE 2 FLOAT
        ARG_TYPE 3 INT
        """
            )
        )
    prog_res_config = ResConfig(
        config={
            "INTERNALS": {},
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "QUEUE_SYSTEM": "LOCAL",
                    "JOBNAME": "SIM_KW",
                },
                "ECLBASE": "SIM_KW",
                "NUM_REALIZATIONS": 10,
                "INSTALL_JOB": [
                    {"NAME": "NEW_JOB_A", "PATH": "NEW_TYPE_A"},
                ],
                "SIMULATION_JOB": [
                    {"NAME": "NEW_JOB_A", "ARGLIST": ["Hello", True, 3.14, 4]},
                    {"NAME": "NEW_JOB_A", "ARGLIST": ["word"]},
                ],
            },
        }
    )
    forward_model = prog_res_config.model_config.getForwardModel()
    job_A = forward_model.iget_job(0)
    job_B = forward_model.iget_job(1)
    assert job_A.get_arglist(), ["Hello", "True", "3.14" == "4"]
    assert job_B.get_arglist() == ["word"]


def test_minimum_config(minimum_example):
    prog_res_config = ResConfig(
        config={
            "INTERNALS": {
                "CONFIG_DIRECTORY": "simple_config",
            },
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "Job%d",
                },
                "RUNPATH": "/tmp/simulations/run%d",
                "NUM_REALIZATIONS": 10,
                "MIN_REALIZATIONS": 10,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
            },
        }
    )

    assert prog_res_config == minimum_example

    assert len(prog_res_config.errors) == 0
    assert len(prog_res_config.failed_keys) == 0


def test_failed_keys(minimum_example):
    res_config = ResConfig(
        config={
            "INTERNALS": {
                "CONFIG_DIRECTORY": "simple_config",
            },
            "UNKNOWN_KEY": "Have/not/got/a/clue",
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "Job%d",
                },
                "RUNPATH": "/tmp/simulations/run%d",
                "NUM_REALIZATIONS": 10,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
            },
        }
    )

    assert len(res_config.failed_keys) == 1
    assert list(res_config.failed_keys.keys()) == ["UNKNOWN_KEY"]
    assert res_config.failed_keys["UNKNOWN_KEY"] == "Have/not/got/a/clue"


@pytest.mark.parametrize(
    "erroring_config",
    [
        {
            "INTERNALS": {
                "CONFIG_DIRECTORY": "simple_config",
            },
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "Job%d",
                },
                "RUNPATH": "/tmp/simulations/run%d",
                "NUM_REALIZATIONS": "/should/be/an/integer",
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
            },
        },
        {
            "INTERNALS": {},
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "Job%d",
                },
                "RUNPATH": "/tmp/simulations/run%d",
                "NUM_REALIZATIONS": 10,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
            },
        },
    ],
)
def test_errors(minimum_example, erroring_config):
    with pytest.raises(ValueError):
        ResConfig(config=erroring_config)


def test_large_config(setup_case):
    res_config = setup_case("local/snake_oil_structure", "user_config.ert")
    prog_res_config = ResConfig(
        config={
            "DEFINES": {
                "<USER>": "TEST_USER",
                "<SCRATCH>": "scratch/ert",
                "<CASE_DIR>": "the_extensive_case",
                "<ECLIPSE_NAME>": "XYZ",
            },
            "INTERNALS": {
                "CONFIG_DIRECTORY": "snake_oil_structure/ert/model",
            },
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "SNAKE_OIL_STRUCTURE_%d",
                    "QUEUE_SYSTEM": "LSF",
                    "MAX_RUNTIME": 23400,
                    "MIN_REALIZATIONS": "50%",
                    "MAX_SUBMIT": 13,
                    "UMASK": "007",
                    "QUEUE_OPTION": [
                        {"DRIVER_NAME": "LSF", "OPTION": "MAX_RUNNING", "VALUE": 100},
                        {
                            "DRIVER_NAME": "LSF",
                            "OPTION": "LSF_RESOURCE",
                            "VALUE": "select[x86_64Linux] same[type:model]",
                        },
                        {
                            "DRIVER_NAME": "LSF",
                            "OPTION": "LSF_SERVER",
                            "VALUE": "simulacrum",
                        },
                        {
                            "DRIVER_NAME": "LSF",
                            "OPTION": "LSF_QUEUE",
                            "VALUE": "mr",
                        },
                    ],
                },
                "DATA_FILE": "../../eclipse/model/SNAKE_OIL.DATA",
                "RUNPATH": "<SCRATCH>/<USER>/<CASE_DIR>/realization-%d/iter-%d",
                "RUNPATH_FILE": "../output/run_path_file/.ert-runpath-list_<CASE_DIR>",
                "ECLBASE": "eclipse/model/<ECLIPSE_NAME>-%d",
                "NUM_REALIZATIONS": "10",
                "ENSPATH": "../output/storage/<CASE_DIR>",
                "GRID": "../../eclipse/include/grid/CASE.EGRID",
                "REFCASE": "../input/refcase/SNAKE_OIL_FIELD",
                "HISTORY_SOURCE": "REFCASE_HISTORY",
                "OBS_CONFIG": "../input/observations/obsfiles/observations.txt",
                "TIME_MAP": "../input/refcase/time_map.txt",
                "SUMMARY": [
                    "WOPR:PROD",
                    "WOPT:PROD",
                    "WWPR:PROD",
                    "WWCT:PROD",
                    "WWPT:PROD",
                    "WBHP:PROD",
                    "WWIR:INJ",
                    "WWIT:INJ",
                    "WBHP:INJ",
                    "ROE:1",
                ],
                "INSTALL_JOB": [
                    {
                        "NAME": "SNAKE_OIL_SIMULATOR",
                        "PATH": "../../snake_oil/jobs/SNAKE_OIL_SIMULATOR",
                    },
                    {
                        "NAME": "SNAKE_OIL_NPV",
                        "PATH": "../../snake_oil/jobs/SNAKE_OIL_NPV",
                    },
                    {
                        "NAME": "SNAKE_OIL_DIFF",
                        "PATH": "../../snake_oil/jobs/SNAKE_OIL_DIFF",
                    },
                ],
                "LOAD_WORKFLOW_JOB": ["../bin/workflows/workflowjobs/UBER_PRINT"],
                "LOAD_WORKFLOW": ["../bin/workflows/MAGIC_PRINT"],
                "FORWARD_MODEL": [
                    "SNAKE_OIL_SIMULATOR",
                    "SNAKE_OIL_NPV",
                    "SNAKE_OIL_DIFF",
                ],
                "RUN_TEMPLATE": ["../input/templates/seed_template.txt", "seed.txt"],
                "GEN_KW": [
                    {
                        "NAME": "SIGMA",
                        "TEMPLATE": "../input/templates/sigma.tmpl",
                        "OUT_FILE": "coarse.sigma",
                        "PARAMETER_FILE": "../input/distributions/sigma.dist",
                    }
                ],
                "GEN_DATA": [
                    {
                        "NAME": "super_data",
                        "RESULT_FILE": "super_data_%d",
                        "REPORT_STEPS": 1,
                    }
                ],
                "LOGGING": {
                    "UPDATE_LOG_PATH": "../output/update_log/<CASE_DIR>",
                },
            },
        }
    )

    assert prog_res_config == res_config
    assert len(prog_res_config.failed_keys) == 0
