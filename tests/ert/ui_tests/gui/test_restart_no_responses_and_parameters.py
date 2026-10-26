import stat
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox

from ert.config import ErtConfig
from ert.gui.experiments.evaluate_ensemble_panel import EvaluateEnsemblePanel
from ert.gui.experiments.experiment_panel import ExperimentPanel
from ert.gui.main_window import ErtMainWindow
from ert.run_models import EnsembleExperiment
from ert.run_models.evaluate_ensemble import EvaluateEnsemble
from ert.storage import Storage
from ert.validation import rangestring_to_mask

from .conftest import _open_main_window as open_main_window
from .conftest import get_child


@contextmanager
def _open_main_window(
    path,
) -> Generator[tuple[ErtMainWindow, Storage, ErtConfig], None, None]:
    fm_model = Path("forward_model.py")
    fm_model.write_text(
        dedent(
            """\
                #!/usr/bin/env python3
                import os

                if __name__ == "__main__":
                    if int(os.getenv("_ERT_REALIZATION_NUMBER")) % 2 == 0:
                        raise ValueError()
                """
        ),
        encoding="utf-8",
    )

    fm_model.chmod(fm_model.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    Path("FORWARD_MODEL").write_text(f"EXECUTABLE {fm_model.name}", encoding="utf-8")

    config = dedent("""
    QUEUE_SYSTEM LOCAL
    QUEUE_OPTION LOCAL MAX_RUNNING 2
    MAX_SUBMIT 1
    NUM_REALIZATIONS 10
    MIN_REALIZATIONS 1
    INSTALL_JOB forward_model FORWARD_MODEL
    FORWARD_MODEL forward_model
    """)
    with Path("config.ert").open("w", encoding="utf-8") as fh:
        fh.writelines(config)

    with open_main_window(path / "config.ert") as opened_window:
        yield opened_window


@pytest.fixture
def open_gui(tmp_path, monkeypatch, run_experiment):
    monkeypatch.chdir(tmp_path)
    with (
        _open_main_window(tmp_path) as (
            gui,
            _,
            __,
        ),
    ):
        yield gui


def test_sensitivity_restart(open_gui, qtbot, run_experiment):
    """This runs a full manual update workflow, first running ensemble experiment
    where some of the realizations fail, then doing an update before running an
    ensemble experiment again to calculate the forecast of the update.
    """
    gui = open_gui
    run_experiment(EnsembleExperiment, gui)
    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_settings = get_child(experiment_panel, EvaluateEnsemblePanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(EvaluateEnsemble.name())

    idx = simulation_settings._ensemble_selector.findData(
        "ensemble_experiment : iter-0",
        Qt.ItemDataRole.DisplayRole,
        Qt.MatchFlag.MatchStartsWith,
    )
    assert idx != -1
    simulation_settings._ensemble_selector.setCurrentIndex(idx)

    storage = gui.notifier.storage
    experiment = storage.get_experiment_by_name("ensemble_experiment")
    ensemble_prior = experiment.get_ensemble_by_name("iter-0")
    success = ensemble_prior.get_realization_mask_without_failure()
    # Assert that some realizations failed
    assert not all(success)
    # Check that the failed realizations are suggested for Evaluate ensemble
    assert list(~success) == rangestring_to_mask(
        experiment_panel.get_experiment_arguments().realizations,
        10,
    )
