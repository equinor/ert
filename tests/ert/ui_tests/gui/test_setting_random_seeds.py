import shutil
from pathlib import Path
from textwrap import dedent

import pytest

from ert.run_models import EnsembleExperiment, SingleTestRun

from .conftest import _open_main_window


@pytest.mark.parametrize("experiment_type", [SingleTestRun, EnsembleExperiment])
def test_that_gui_uses_config_random_seed_when_specified(
    run_experiment, use_tmpdir, qtbot, experiment_type
):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        RANDOM_SEED 12345

        QUEUE_SYSTEM LOCAL
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    with _open_main_window("config.ert") as (gui, _, _):
        run_experiment(experiment_type, gui)
        experiments = list(gui.notifier.storage.experiments)
        assert len(experiments) == 1
        assert experiments[0].experiment_config["random_seed"] == 12345


@pytest.mark.parametrize("experiment_type", [SingleTestRun, EnsembleExperiment])
def test_that_gui_generates_different_seeds_for_consecutive_runs(
    run_experiment, use_tmpdir, qtbot, experiment_type
):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        RUNPATH gui_random_seed/realization-<IENS>/iter-<ITER>

        QUEUE_SYSTEM LOCAL
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    with _open_main_window("config.ert") as (gui, _, _):
        run_experiment(experiment_type, gui)

        # run_experiment expects the runpath to not exist
        shutil.rmtree("gui_random_seed")

        run_experiment(experiment_type, gui)
        experiments = list(gui.notifier.storage.experiments)
        assert len(experiments) == 2
        assert (
            len(
                {
                    experiment.experiment_config["random_seed"]
                    for experiment in experiments
                }
            )
            == 2
        )
