import os
from argparse import Namespace

import pytest

from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert.shared.cli.workflow import execute_workflow
from ert.shared.plugins.plugin_manager import ErtPluginContext


@pytest.mark.usefixtures("copy_poly_case")
def test_executing_workflow():
    with ErtPluginContext():
        with open("test_wf", "w") as wf_file:
            wf_file.write("EXPORT_RUNPATH")

        config_file = "poly.ert"
        with open(config_file, "a") as file:
            file.write("LOAD_WORKFLOW test_wf")

        rc = ResConfig(user_config_file=config_file)
        ert = EnKFMain(rc)
        args = Namespace(name="test_wf")
        execute_workflow(ert, args.name)
        assert os.path.isfile(".ert_runpath_list")
