from pathlib import Path
from textwrap import dedent

import pytest

from ert.config import ConfigWarning, ErtConfig
from ert.plugins import ErtPluginContext


def test_that_ert_warns_on_duplicate_workflow_jobs(tmp_path):
    """
    Tests that we emit a ConfigWarning if we detect multiple
    workflows with the same name during config parsing.
    Relies on the internal workflow CAREFUL_COPY_FILE.
    """
    test_workflow_job = tmp_path / "CAREFUL_COPY_FILE"
    with open(test_workflow_job, "w", encoding="utf-8") as fh:
        fh.write("EXECUTABLE test_copy_duplicate.py")
    test_workflow_job_executable = tmp_path / "test_copy_duplicate.py"
    Path(test_workflow_job_executable).touch(mode=0o755)
    test_config_file_name = tmp_path / "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        LOAD_WORKFLOW_JOB CAREFUL_COPY_FILE
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with (
        pytest.warns(
            ConfigWarning, match="Duplicate workflow jobs with name 'CAREFUL_COPY_FILE'"
        ),
        ErtPluginContext(),
    ):
        _ = ErtConfig.from_file(test_config_file_name)
