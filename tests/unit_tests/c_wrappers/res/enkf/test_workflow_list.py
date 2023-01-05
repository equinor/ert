import os
from textwrap import dedent

import pytest

from ert._c_wrappers.enkf import ConfigKeys, ErtWorkflowList, ResConfig
from ert._c_wrappers.enkf.enums import HookRuntime
from ert._c_wrappers.enkf.res_config import site_config_location
from ert._c_wrappers.job_queue.workflow_job import WorkflowJob


@pytest.mark.usefixtures("copy_minimum_case")
def test_workflow_list_constructor():
    ERT_SITE_CONFIG = site_config_location()
    ERT_SHARE_PATH = os.path.dirname(ERT_SITE_CONFIG)
    cwd = os.getcwd()

    config_dict = {
        ConfigKeys.LOAD_WORKFLOW_JOB: [
            [cwd + "/workflows/UBER_PRINT", "print_uber"],
            [cwd + "/workflows/HIDDEN_PRINT", "HIDDEN_PRINT"],
        ],
        ConfigKeys.LOAD_WORKFLOW: [
            [cwd + "/workflows/MAGIC_PRINT", "magic_print"],
            [cwd + "/workflows/NO_PRINT", "no_print"],
            [cwd + "/workflows/SOME_PRINT", "some_print"],
        ],
        ConfigKeys.WORKFLOW_JOB_DIRECTORY: [
            ERT_SHARE_PATH + "/workflows/jobs/shell",
            ERT_SHARE_PATH + "/workflows/jobs/internal/config",
            ERT_SHARE_PATH + "/workflows/jobs/internal-gui/config",
        ],
        ConfigKeys.HOOK_WORKFLOW_KEY: [
            ["magic_print", "POST_UPDATE"],
            ["no_print", "PRE_UPDATE"],
        ],
    }

    with open("minimum_config", "a+", encoding="utf-8") as ert_file:
        ert_file.write("LOAD_WORKFLOW_JOB workflows/UBER_PRINT print_uber\n")
        ert_file.write("LOAD_WORKFLOW_JOB workflows/HIDDEN_PRINT\n")
        ert_file.write("LOAD_WORKFLOW workflows/MAGIC_PRINT magic_print\n")
        ert_file.write("LOAD_WORKFLOW workflows/NO_PRINT no_print\n")
        ert_file.write("LOAD_WORKFLOW workflows/SOME_PRINT some_print\n")
        ert_file.write("HOOK_WORKFLOW magic_print POST_UPDATE\n")
        ert_file.write("HOOK_WORKFLOW no_print PRE_UPDATE\n")

    os.mkdir("workflows")

    with open("workflows/MAGIC_PRINT", "w", encoding="utf-8") as f:
        f.write("print_uber\n")
    with open("workflows/NO_PRINT", "w", encoding="utf-8") as f:
        f.write("print_uber\n")
    with open("workflows/SOME_PRINT", "w", encoding="utf-8") as f:
        f.write("print_uber\n")
    with open("workflows/UBER_PRINT", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE ls\n")
    with open("workflows/HIDDEN_PRINT", "w", encoding="utf-8") as f:
        f.write("EXECUTABLE ls\n")

    res_config = ResConfig("minimum_config")
    ert_workflow_list = ErtWorkflowList.from_dict(config_dict)

    assert ert_workflow_list.getJobNames() == res_config.ert_workflow_list.getJobNames()

    # verify name generated from filename
    assert "HIDDEN_PRINT" in ert_workflow_list.getJobNames()
    assert "print_uber" in ert_workflow_list.getJobNames()

    assert [
        "magic_print",
        "no_print",
        "some_print",
    ] == ert_workflow_list.getWorkflowNames()

    assert (
        len(list(ert_workflow_list.get_workflows_hooked_at(HookRuntime.PRE_UPDATE)))
        == 1
    )
    assert (
        len(list(ert_workflow_list.get_workflows_hooked_at(HookRuntime.POST_UPDATE)))
        == 1
    )
    assert (
        len(
            list(
                ert_workflow_list.get_workflows_hooked_at(HookRuntime.PRE_FIRST_UPDATE)
            )
        )
        == 0
    )

    assert ert_workflow_list == res_config.ert_workflow_list


@pytest.mark.usefixtures("use_tmpdir")
def test_job_load_OK():
    script_file_contents = dedent(
        """
        INTERNAL   False
        EXECUTABLE /usr/bin/python
        MIN_ARG    1
        MAX_ARG    2
        ARG_TYPE   0  STRING
        ARG_TYPE   1  INT
        """
    )

    script_file_path = os.path.join(os.getcwd(), "externalOK")
    with open(script_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(script_file_contents)

    WorkflowJob.fromFile(script_file_path)
