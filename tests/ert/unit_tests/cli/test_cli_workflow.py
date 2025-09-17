import os
from argparse import Namespace

import pytest

from ert.cli.workflow import execute_workflow
from ert.config import ErtConfig
from ert.plugins.plugin_manager import ErtPluginContext


@pytest.mark.usefixtures("copy_poly_case")
def test_executing_workflow(storage):
    with ErtPluginContext() as ctx:
        with open("test_wf", "w", encoding="utf-8") as wf_file:
            wf_file.write("CSV_EXPORT test_workflow_output.csv")

        config_file = "poly.ert"
        with open(config_file, "a", encoding="utf-8") as file_handle:
            file_handle.write("LOAD_WORKFLOW test_wf")

        rc = ErtConfig.with_plugins(ctx).from_file(config_file)
        args = Namespace(name="test_wf")
        execute_workflow(rc, storage, args.name)
        assert os.path.isfile("test_workflow_output.csv")
