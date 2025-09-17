import stat
from contextlib import suppress
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest

from ert.cli.main import ErtCliError
from ert.plugins import ErtPluginContext

from .run_cli import run_cli_with_pm

config_contents = """\
QUEUE_SYSTEM {queue_system}
NUM_REALIZATIONS 10
LOAD_WORKFLOW_JOB CHMOD_JOB CHMOD
LOAD_WORKFLOW CHMOD.wf CHMOD.wf
HOOK_WORKFLOW CHMOD.wf PRE_SIMULATION

"""

workflow_contents = """\
CHMOD
"""

workflow_job_contents = """\
EXECUTABLE chmod.sh
"""

chmod_sh_contents = """\
#!/bin/bash
chmod 000 {tmp_path}/simulations/realization-0/iter-0
"""


def write_config(tmp_path, queue_system):
    (tmp_path / "config.ert").write_text(
        config_contents.format(queue_system=queue_system)
    )
    (tmp_path / "CHMOD_JOB").write_text(workflow_job_contents)
    (tmp_path / "CHMOD.wf").write_text(workflow_contents)
    (tmp_path / "chmod.sh").write_text(chmod_sh_contents.format(tmp_path=tmp_path))
    (tmp_path / "chmod.sh").chmod(
        (tmp_path / "chmod.sh").stat().st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )


def test_missing_runpath_has_isolated_failures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path, "LOCAL")
    try:
        with pytest.raises(
            ErtCliError,
            match=r"successful realizations \(9\) is less "
            r"than .* MIN_REALIZATIONS\(10\)",
        ):
            run_cli_with_pm(
                ["ensemble_experiment", "config.ert", "--disable-monitoring"]
            )
    finally:
        with suppress(FileNotFoundError):
            (tmp_path / "simulations/realization-0/iter-0").chmod(0x777)


def raising_named_temporary_file(*args, **kwargs):
    if "realization-1" in str(kwargs["dir"]):
        raise OSError("Don't like realization-1")
    return NamedTemporaryFile(*args, **kwargs)


def patch_raising_named_temporary_file(queue_system):
    return patch(
        f"ert.scheduler.{queue_system}_driver.NamedTemporaryFile",
        raising_named_temporary_file,
    )


def test_failing_writes_lead_to_isolated_failures(tmp_path, monkeypatch, pytestconfig):
    monkeypatch.chdir(tmp_path)
    queue_system = None
    if pytestconfig.getoption("lsf"):
        # queue_system = "LSF"
        pytest.skip(reason="Currently does not work with the lsf setup")
    elif pytestconfig.getoption("slurm"):
        queue_system = "SLURM"
    else:
        pytest.skip()
    (tmp_path / "config.ert").write_text(
        f"""
        QUEUE_SYSTEM {queue_system}
        NUM_REALIZATIONS 10
        """
    )
    with (
        pytest.raises(
            ErtCliError,
            match=r"(?s)successful realizations \(9\) is less "
            r"than .* MIN_REALIZATIONS\(10\).*"
            "Driver reported: Could not create"
            " submit script: Don't like realization-1",
        ),
        patch_raising_named_temporary_file(queue_system.lower()),
        ErtPluginContext() as runtime_plugins,
    ):
        run_cli_with_pm(
            ["ensemble_experiment", "config.ert", "--disable-monitoring"],
            runtime_plugins=runtime_plugins,
        )
