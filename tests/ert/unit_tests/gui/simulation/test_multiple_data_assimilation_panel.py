from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
from PyQt6.QtWidgets import QCheckBox
from pytestqt.qtbot import QtBot

from ert.config.analysis_config import AnalysisConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector
from ert.gui.ertwidgets.stringbox import StringBox
from ert.gui.simulation.multiple_data_assimilation_panel import (
    MultipleDataAssimilationPanel,
)
from ert.storage.local_ensemble import LocalEnsemble
from ert.storage.local_experiment import LocalExperiment
from ert.storage.local_storage import LocalStorage
from ert.storage.realization_storage_state import RealizationStorageState


class MockStorage(LocalStorage):
    def __init__(self) -> None:
        self._ensembles = {}
        self._experiments = {}

    def _setup_mocked_previous_run(self) -> None:
        mock_ensemble = MagicMock(spec=LocalEnsemble)
        mock_ensemble.id = uuid4()
        mock_ensemble.name = "mock_ensemble"
        mock_ensemble.relative_weights = "4, 2, 1"

        def mock_get_ensemble_state(*args, **kwargs):
            realization_finished_successfully = {
                RealizationStorageState.PARAMETERS_LOADED,
                RealizationStorageState.RESPONSES_LOADED,
            }
            realization_failed = {RealizationStorageState.UNDEFINED}
            # Working realization range string should be '0-2, 5'
            return [
                realization_finished_successfully,
                realization_finished_successfully,
                realization_finished_successfully,
                realization_failed,
                realization_failed,
                realization_finished_successfully,
            ]

        mock_ensemble.get_ensemble_state = mock_get_ensemble_state

        def mock_get_realization_mask_with_responses(*args):
            return np.array(
                [
                    RealizationStorageState.RESPONSES_LOADED in state
                    for state in mock_ensemble.get_ensemble_state()
                ]
            )

        mock_ensemble.get_realization_mask_with_responses = (
            mock_get_realization_mask_with_responses
        )
        mock_experiment = MagicMock(spec=LocalExperiment)
        mock_experiment.ensembles = [mock_ensemble]
        mock_experiment.id = uuid4()
        mock_experiment.name = "mock_experiment"

        mock_ensemble.experiment_id = mock_experiment.id
        mock_ensemble.experiment = mock_experiment
        self._ensembles[mock_ensemble.id] = mock_ensemble
        self._experiments[mock_experiment.id] = mock_experiment


def setup_notifier() -> ErtNotifier:
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    return notifier


@pytest.mark.usefixtures("copy_poly_case")
def test_that_active_realizations_selector_validates_with_ensemble_size_from_config(
    qtbot: QtBot,
) -> None:
    """This is a test that makes sure the realization selector autofills and
    validates with the num_realizations from config/designmatrix, and the autofilled
    configuration is valid. It also makes sure the restart run button is disabled if
    there are no previous experiments/ensembles in storage"""
    active_realizations = [True, True, False, True, False, True, True]
    config_num_realizations = len(active_realizations)
    panel = MultipleDataAssimilationPanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        run_path="",
        notifier=setup_notifier(),
        active_realizations=active_realizations,
        config_num_realization=config_num_realizations,
    )
    qtbot.addWidget(panel)
    realization_selector = panel.findChild(StringBox, "active_realizations_box")
    ensemble_selector = panel.findChild(EnsembleSelector)
    assert realization_selector is not None
    assert realization_selector.isValid()
    assert realization_selector.get_text == "0-1, 3, 5-6"
    assert not ensemble_selector.isEnabled()
    restart_checkbox = panel.findChild(QCheckBox, "restart_checkbox_esmda")
    assert not restart_checkbox.isEnabled()
    assert not restart_checkbox.isChecked()
    assert panel.isConfigurationValid()


@pytest.mark.usefixtures("copy_poly_case")
def test_that_active_realizations_selector_validates_with_with_realizations_from_storage_on_rerun_from(  # noqa: E501
    qtbot: QtBot,
) -> None:
    """This is a test that makes sure that the active realizations field is
    validated against the num_realizations from config/designmatrix on default,
    but swaps to the realizations found in storage if the
    'rerun from' checkbox is toggled
    """

    config_num_realizations = 20
    active_realizations = [True] * config_num_realizations
    notifier = setup_notifier()
    notifier._storage._setup_mocked_previous_run()
    panel = MultipleDataAssimilationPanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        run_path="",
        notifier=notifier,
        active_realizations=active_realizations,
        config_num_realization=config_num_realizations,
    )

    qtbot.addWidget(panel)
    realization_selector = panel.findChild(StringBox, "active_realizations_box")
    ensemble_selector = panel.findChild(EnsembleSelector)
    assert realization_selector is not None
    assert realization_selector.isValid()
    assert realization_selector.text() == "0-19"
    assert panel.isConfigurationValid()
    assert not ensemble_selector.isEnabled()
    restart_checkbox = panel.findChild(QCheckBox, "restart_checkbox_esmda")
    assert restart_checkbox.isEnabled()
    assert not restart_checkbox.isChecked()
    restart_checkbox.click()
    assert restart_checkbox.isChecked()
    assert ensemble_selector.isEnabled()
    assert ensemble_selector.currentText() == "mock_experiment : mock_ensemble"
    assert realization_selector.text() == "0-2, 5"
    assert panel.isConfigurationValid()
    # We try rerunning from a realization that failed previously
    realization_selector.setText("0-2, 4-5")
    assert not panel.isConfigurationValid()
    restart_checkbox.setChecked(False)
    assert realization_selector.text() == "0-19"
    assert panel.isConfigurationValid()
