import os
from textwrap import dedent

import pytest

from ert.cli.main import ErtCliError
from ert.shared.plugins import ErtPluginContext
from tests.integration_tests.run_cli import run_cli_with_pm


@pytest.fixture()
def queue_name_config():
    if queue_name := os.getenv("_ERT_TESTS_DEFAULT_QUEUE_NAME"):
        return f"QUEUE_OPTION LSF LSF_QUEUE {queue_name}\n"
    return ""


@pytest.mark.out_of_memory
@pytest.mark.requires_lsf
@pytest.mark.timeout(600)
def test_that_oom_kills_are_reported(queue_name_config, tmp_path):
    """This test will do a local test run that will run a job that consumes all
    memory until it is killed by the operating system. The test will assert
    that Ert is able to pick up what happen and present the cause (oom-killer)
    to the user).

    This test is not run by default, but requires --runoom supplied on the
    command line to pytest, as it will take a long time and make the system
    partially unresponsive for all logged in users (more swap will make the
    time of unresponsiveness longer)."""

    print(os.getcwd())

    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\n"
        "\nINSTALL_JOB memory_hog MEMORY_HOG\nSIMULATION_JOB memory_hog\n"
        + queue_name_config
        + "QUEUE_SYSTEM LSF\n"
    )

    (tmp_path / "MEMORY_HOG").write_text(
        f"EXECUTABLE {tmp_path / 'memory_hog.sh'}", encoding="utf-8"
    )
    (tmp_path / "memory_hog.sh").write_text(
        dedent(
            """#!/bin/sh
    perl -wE 'my @xs; for (1..2**20) { push @xs, q{a} x 2**20 }; say scalar @xs;'
    """
        ),
        encoding="utf-8",
    )
    os.chmod(tmp_path / "memory_hog.sh", 0o0755)
    with pytest.raises(
        ErtCliError,
        match="Realization.*0 failed.*memory_hog.*was killed due to out-of-memory",
    ), ErtPluginContext() as context:
        run_cli_with_pm(
            ["ensemble_experiment", str(tmp_path / "config.ert")],
            context.plugin_manager,
        )
