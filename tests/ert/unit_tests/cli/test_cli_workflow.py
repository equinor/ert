import logging
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
    with Path(config_file).open("a", encoding="utf-8") as file_handle:
        file_handle.write("LOAD_WORKFLOW test_wf")

    rc = ErtConfig.with_plugins(get_site_plugins()).from_file(config_file)
    args = Namespace(name="test_wf")
    execute_workflow(rc, storage, args.name)
    assert Path("test_workflow_output.csv").is_file()


@pytest.mark.usefixtures("copy_poly_case")
def test_that_output_of_workflow_run_from_cli_is_in_ert_log(storage, caplog):
    Path("print_job").write_text("EXECUTABLE print_script.sh\n", encoding="utf-8")
    print_script = Path("print_script.sh")
    print_script.write_text(
        "#!/bin/bash\necho hello from the cli workflow\n", encoding="utf-8"
    )
    print_script.chmod(print_script.stat().st_mode | 0o111)
    Path("print_workflow").write_text("printjob\n", encoding="utf-8")

    config_file = "poly.ert"
    with Path(config_file).open("a", encoding="utf-8") as file_handle:
        file_handle.write(
            "LOAD_WORKFLOW_JOB print_job printjob\n"
            "LOAD_WORKFLOW print_workflow wfprint\n"
        )

    rc = ErtConfig.with_plugins(get_site_plugins()).from_file(config_file)

    with caplog.at_level(logging.INFO, logger="ert.workflow_runner"):
        execute_workflow(rc, storage, "wfprint")

    assert "workflow=wfprint job=printjob#0 status=success" in caplog.text
    assert "--- stdout ---\nhello from the cli workflow" in caplog.text
