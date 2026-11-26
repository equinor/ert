from pathlib import Path
from textwrap import dedent

import pytest

from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    TEST_RUN_MODE,
)

from .run_cli import read_server_log, run_cli


def server_log_lines_containing(text: str) -> list[str]:
    return [line for line in read_server_log().splitlines() if text in line]


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize("experiment_type", [TEST_RUN_MODE, ENSEMBLE_EXPERIMENT_MODE])
def test_that_cli_uses_config_random_seed_when_specified(use_tmpdir, experiment_type):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        RANDOM_SEED 12345
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    run_cli(
        experiment_type,
        "--disable-monitoring",
        "config.ert",
    )

    seed_logs = server_log_lines_containing("'random_seed':")
    assert len(seed_logs) == 1
    assert "'random_seed': 12345" in seed_logs[0]


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize("experiment_type", [TEST_RUN_MODE, ENSEMBLE_EXPERIMENT_MODE])
def test_that_cli_generates_different_seeds_for_consecutive_runs(
    use_tmpdir, experiment_type
):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    run_cli(
        experiment_type,
        "--disable-monitoring",
        "config.ert",
    )
    first_seed_from_log = server_log_lines_containing("RANDOM_SEED")[-1]

    run_cli(
        experiment_type,
        "--disable-monitoring",
        "config.ert",
    )
    second_seed_from_log = server_log_lines_containing("RANDOM_SEED")[-1]

    assert first_seed_from_log != second_seed_from_log

    assert len(server_log_lines_containing("'random_seed':")) == 2
