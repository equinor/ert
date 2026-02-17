from PyQt6.QtCore import Qt
from pytestqt.qtbot import QtBot

from ert.config.analysis_config import AnalysisConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import EnsembleSelector, StringBox
from ert.gui.experiments.manual_update_panel import ManualUpdatePanel

from .conftest import (
    REALIZATION_FINISHED_SUCCESSFULLY,
    REALIZATION_UNDEFINED,
    MockStorage,
)


def test_that_active_realizations_selector_validates_with_ensemble_size_from_prior(
    qtbot: QtBot,
) -> None:
    """This is a test that makes sure that the active realizations field is
    validated against the ensemble size from the prior ensemble, and not
    the ensemble size from config.
    """
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    notifier._storage._setup_mocked_run(
        "mock_ensemble0",
        "mock_experiment0",
        [
            REALIZATION_UNDEFINED,
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_UNDEFINED,
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_FINISHED_SUCCESSFULLY,
        ],
    )
    notifier._storage._setup_mocked_run(
        "mock_ensemble1",
        "mock_experiment1",
        [
            REALIZATION_UNDEFINED,
            REALIZATION_UNDEFINED,
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_UNDEFINED,
            REALIZATION_FINISHED_SUCCESSFULLY,
        ],
    )
    panel = ManualUpdatePanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        run_path="",
        notifier=notifier,
    )
    qtbot.addWidget(panel)

    prior_ensemble0_ensemble_size = 8
    prior_ensemble1_ensemble_size = 6

    realization_selector = panel.findChild(StringBox, "active_realizations_box")
    assert realization_selector is not None
    ensemble_selector = panel.findChild(EnsembleSelector)
    assert ensemble_selector.isEnabled()

    index = ensemble_selector.findText("mock_ensemble0", Qt.MatchFlag.MatchContains)
    ensemble_selector.setCurrentIndex(index)
    assert ensemble_selector.currentText() == "mock_experiment0 : mock_ensemble0"
    assert (
        panel._number_of_realizations_label.text()
        == f"<b>{prior_ensemble0_ensemble_size}</b>"
    )

    # Only these realizations in prior have RealizationStorageState.RESPONSES_LOADED
    assert realization_selector.text() == "1-3, 5-7"
    assert realization_selector.isValid(), (
        realization_selector._validation._validation_message
    )
    assert panel.isConfigurationValid()
    assert (
        panel.get_experiment_arguments().ensemble_size == prior_ensemble0_ensemble_size
    )

    # We try running a realization that does not have RESPONSES_LOADED
    realization_selector.setText("4-7")
    assert not panel.isConfigurationValid()
    index = ensemble_selector.findText("mock_ensemble1", Qt.MatchFlag.MatchContains)
    ensemble_selector.setCurrentIndex(index)
    assert ensemble_selector.currentText() == "mock_experiment1 : mock_ensemble1"

    # The active realizations field should auto-populate with a valid value
    assert panel.isConfigurationValid()
    assert (
        panel.get_experiment_arguments().ensemble_size == prior_ensemble1_ensemble_size
    )
    assert (
        panel._number_of_realizations_label.text()
        == f"<b>{prior_ensemble1_ensemble_size}</b>"
    )
    assert realization_selector.text() == "2-3, 5"


def test_that_manual_update_ensemble_selector_only_shows_ensembles_with_data(
    qtbot: QtBot,
) -> None:
    """This is a test that makes sure that ensembles without data are not available
    for selection in the ensemble_selector. This will be ensembles that were just
    created by ManageExperiment or by ManualUpdate.
    """
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    notifier._storage._setup_mocked_run(
        "mock_ensemble_no_data",
        "mock_experiment2",
        [REALIZATION_UNDEFINED, REALIZATION_UNDEFINED, REALIZATION_UNDEFINED],
    )

    panel = ManualUpdatePanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        run_path="",
        notifier=notifier,
    )
    qtbot.addWidget(panel)
    ensemble_selector = panel.findChild(EnsembleSelector)
    assert not ensemble_selector.isEnabled()

    index = ensemble_selector.findText(
        "mock_ensemble_no_data", Qt.MatchFlag.MatchContains
    )
    assert index == -1  # Invalid index, because it is not available
    ensemble_selector.setCurrentIndex(index)
    assert not ensemble_selector.currentText()
    assert not panel.isConfigurationValid()
