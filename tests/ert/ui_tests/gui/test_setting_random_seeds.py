import logging
import shutil
from pathlib import Path
from textwrap import dedent

import pytest

from ert.run_models import EnsembleExperiment, SingleTestRun

from .conftest import _open_main_window


@pytest.mark.parametrize("experiment_type", [SingleTestRun, EnsembleExperiment])
def test_that_gui_uses_config_random_seed_when_specified(
    run_experiment, use_tmpdir, qtbot, caplog, experiment_type
):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        RANDOM_SEED 12345
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    with (
        caplog.at_level(logging.INFO),
        _open_main_window("config.ert") as (gui, _, _),
    ):
        run_experiment(experiment_type, gui)

    seed_logs = [line for line in caplog.text.splitlines() if "'random_seed':" in line]
    assert len(seed_logs) == 1
    assert "'random_seed': 12345" in seed_logs[0]


@pytest.mark.parametrize("experiment_type", [SingleTestRun, EnsembleExperiment])
def test_that_gui_generates_different_seeds_for_consecutive_runs(
    run_experiment, use_tmpdir, qtbot, caplog, experiment_type
):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        RUNPATH gui_random_seed/realization-<IENS>/iter-<ITER>
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    with (
        caplog.at_level(logging.INFO),
        _open_main_window("config.ert") as (gui, _, _),
    ):
        run_experiment(experiment_type, gui)
        seed_logs = [line for line in caplog.text.splitlines() if "RANDOM_SEED" in line]
        first_seed_from_log = seed_logs[-1]

        # run_experiment expects the runpath to not exist
        shutil.rmtree("gui_random_seed")

        run_experiment(experiment_type, gui)
        seed_logs = [line for line in caplog.text.splitlines() if "RANDOM_SEED" in line]
        second_seed_from_log = seed_logs[-1]

    assert first_seed_from_log != second_seed_from_log

    seed_logs = [line for line in caplog.text.splitlines() if "'random_seed':" in line]
    assert len(seed_logs) == 2
