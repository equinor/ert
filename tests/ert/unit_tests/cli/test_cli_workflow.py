import os
from argparse import Namespace
from pathlib import Path

import pytest

from ert.cli.workflow import execute_workflow
from ert.config import ErtConfig
from ert.plugins import get_site_plugins


@pytest.mark.usefixtures("copy_poly_case")
def test_executing_workflow(storage):
    Path("test_wf").write_text("CSV_EXPORT test_workflow_output.csv", encoding="utf-8")

    config_file = "poly.ert"
    with open(config_file, "a", encoding="utf-8") as file_handle:
        file_handle.write("LOAD_WORKFLOW test_wf")

    rc = ErtConfig.with_plugins(get_site_plugins()).from_file(config_file)
    args = Namespace(name="test_wf")
    execute_workflow(rc, storage, args.name)
    assert os.path.isfile("test_workflow_output.csv")
