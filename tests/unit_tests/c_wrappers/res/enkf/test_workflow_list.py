import os

import pytest

from ert._c_wrappers.enkf import ConfigKeys, ErtWorkflowList, ResConfig
from ert._c_wrappers.enkf.res_config import site_config_location


@pytest.mark.usefixtures("copy_minimum_case")
def test_workflow_list_constructor():
    ERT_SITE_CONFIG = site_config_location()
    ERT_SHARE_PATH = os.path.dirname(ERT_SITE_CONFIG)

    config_dict = {
        ConfigKeys.LOAD_WORKFLOW_JOB: [
            {
                ConfigKeys.NAME: "print_uber",
                ConfigKeys.PATH: os.getcwd() + "/workflows/UBER_PRINT",
            }
        ],
        ConfigKeys.LOAD_WORKFLOW: [
            {
                ConfigKeys.NAME: "magic_print",
                ConfigKeys.PATH: os.getcwd() + "/workflows/MAGIC_PRINT",
            }
        ],
        ConfigKeys.WORKFLOW_JOB_DIRECTORY: [
            ERT_SHARE_PATH + "/workflows/jobs/shell",
            ERT_SHARE_PATH + "/workflows/jobs/internal/config",
            ERT_SHARE_PATH + "/workflows/jobs/internal-gui/config",
        ],
    }

    with open("minimum_config", "a+") as ert_file:
        ert_file.write("LOAD_WORKFLOW_JOB workflows/UBER_PRINT print_uber\n")
        ert_file.write("LOAD_WORKFLOW workflows/MAGIC_PRINT magic_print\n")

    os.mkdir("workflows")

    with open("workflows/MAGIC_PRINT", "w") as f:
        f.write("print_uber\n")
    with open("workflows/UBER_PRINT", "w") as f:
        f.write("EXECUTABLE ls\n")

    res_config = ResConfig("minimum_config")

    assert (
        ErtWorkflowList(
            subst_list=res_config.subst_config.subst_list,
            config_dict=config_dict,
        )
        == res_config.ert_workflow_list
    )


def test_illegal_configs():
    with pytest.raises(ValueError):
        ErtWorkflowList(subst_list=[], config_dict=[], config_content=[])

    with pytest.raises(ValueError):
        ErtWorkflowList()
