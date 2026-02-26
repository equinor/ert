import logging
from pathlib import Path
from textwrap import dedent

import pytest

from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    TEST_RUN_MODE,
)

from .run_cli import run_cli


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize("experiment_type", [TEST_RUN_MODE, ENSEMBLE_EXPERIMENT_MODE])
def test_that_cli_uses_config_random_seed_when_specified(
    caplog, use_tmpdir, experiment_type
):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        RANDOM_SEED 12345
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    with caplog.at_level(logging.INFO):
        run_cli(
            experiment_type,
            "--disable-monitoring",
            "config.ert",
        )

    seed_logs = [line for line in caplog.text.splitlines() if "'random_seed':" in line]
    assert len(seed_logs) == 1
    assert "'random_seed': 12345" in seed_logs[0]


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize("experiment_type", [TEST_RUN_MODE, ENSEMBLE_EXPERIMENT_MODE])
def test_that_cli_generates_different_seeds_for_consecutive_runs(
    caplog, use_tmpdir, experiment_type
):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    with caplog.at_level(logging.INFO):
        run_cli(
            experiment_type,
            "--disable-monitoring",
            "config.ert",
        )
        seed_logs = [line for line in caplog.text.splitlines() if "RANDOM_SEED" in line]
        first_seed_from_log = seed_logs[-1]

        run_cli(
            experiment_type,
            "--disable-monitoring",
            "config.ert",
        )
        seed_logs = [line for line in caplog.text.splitlines() if "RANDOM_SEED" in line]
        second_seed_from_log = seed_logs[-1]

    assert first_seed_from_log != second_seed_from_log

    seed_logs = [line for line in caplog.text.splitlines() if "'random_seed':" in line]
    assert len(seed_logs) == 2
