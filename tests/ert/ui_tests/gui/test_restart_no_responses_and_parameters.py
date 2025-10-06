import os
import stat
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox

from ert.config import ErtConfig
from ert.gui.main import _setup_main_window
from ert.gui.main_window import ErtMainWindow
from ert.gui.simulation.evaluate_ensemble_panel import EvaluateEnsemblePanel
from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.tools.event_viewer import GUILogHandler
from ert.run_models import EnsembleExperiment
from ert.run_models.evaluate_ensemble import EvaluateEnsemble
from ert.storage import Storage
from ert.validation import rangestring_to_mask

from .conftest import get_child


@contextmanager
def _open_main_window(
    path,
) -> Generator[tuple[ErtMainWindow, Storage, ErtConfig], None, None]:
    Path("forward_model.py").write_text(
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

    os.chmod(
        "forward_model.py",
        os.stat("forward_model.py").st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH,
    )
    Path("FORWARD_MODEL").write_text("EXECUTABLE forward_model.py", encoding="utf-8")

    config = dedent("""
    QUEUE_SYSTEM LOCAL
    QUEUE_OPTION LOCAL MAX_RUNNING 2
    MAX_SUBMIT 1
    NUM_REALIZATIONS 10
    MIN_REALIZATIONS 1
    INSTALL_JOB forward_model FORWARD_MODEL
    FORWARD_MODEL forward_model
    """)
    with open("config.ert", "w", encoding="utf-8") as fh:
        fh.writelines(config)

    config = ErtConfig.from_file(path / "config.ert")

    args_mock = Mock()
    args_mock.config = "config.ert"
    # handler defined here to ensure lifetime until end of function, if inlined
    # it will cause the following error:
    # RuntimeError: wrapped C/C++ object of type GUILogHandler
    handler = GUILogHandler()
    gui = _setup_main_window(config, args_mock, handler, config.ens_path)
    yield gui, config.ens_path, config
    gui.close()


@pytest.fixture
def open_gui(tmp_path, monkeypatch, run_experiment, tmp_path_factory):
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
