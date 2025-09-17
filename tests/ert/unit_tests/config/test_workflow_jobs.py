from pathlib import Path
from textwrap import dedent

import pytest

from ert.config import ConfigValidationError, ConfigWarning, ErtConfig
from ert.config.workflow_job import workflow_job_from_file
from ert.plugins import ErtPluginContext


def test_reading_non_existent_workflow_job_raises_config_error():
    with pytest.raises(ConfigValidationError, match="No such file or directory"):
        workflow_job_from_file("/tmp/does_not_exist")


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
        ErtPluginContext() as runtime_plugins,
    ):
        _ = ErtConfig.with_plugins(runtime_plugins).from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_stop_on_fail_is_parsed_external():
    with open("fail_job", "w+", encoding="utf-8") as f:
        f.write("EXECUTABLE echo\n")
        f.write("MIN_ARG 1\n")
        f.write("STOP_ON_FAIL True\n")

    job_internal = workflow_job_from_file(
        name="FAIL",
        config_file="fail_job",
    )

    assert job_internal.stop_on_fail
