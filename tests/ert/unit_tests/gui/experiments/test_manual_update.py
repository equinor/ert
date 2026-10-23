from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QPushButton,
)
from pytestqt.qtbot import QtBot

from ert.config import GenKwConfig
from ert.config.analysis_config import AnalysisConfig
from ert.config.distribution import RawSettings
from ert.config.parameter_config import LocalizationType
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import EnsembleSelector, StringBox
from ert.gui.experiments._update_strategy_summary_widget import (
    UpdateStrategySummaryWidget,
)
from ert.gui.experiments.manual_update_panel import ManualUpdatePanel
from tests.ert.unit_tests.gui.conftest import click_dialog_button

from .conftest import (
    REALIZATION_FINISHED_SUCCESSFULLY,
    REALIZATION_ONLY_RESPONSES,
    REALIZATION_UNDEFINED,
    MockStorage,
)


def test_that_active_realizations_selector_validates_with_ensemble_size_from_prior(
    qtbot: QtBot,
):
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
        runpath="",
        notifier=notifier,
        parameter_configuration=[],
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
):
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
        runpath="",
        notifier=notifier,
        parameter_configuration=[],
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


def test_that_panel_does_not_crash_when_no_realization_has_parameters(qtbot: QtBot):
    """Regression test for https://github.com/equinor/ert/issues/13194.

    When an ensemble has responses loaded but no parameters loaded, the active
    realizations mask is all-False, which converts to an empty rangestring.
    The panel must not crash in this case.
    """
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    notifier._storage._setup_mocked_run(
        "ensemble_with_responses_only",
        "experiment_responses_only",
        [REALIZATION_ONLY_RESPONSES, REALIZATION_ONLY_RESPONSES],
    )
    panel = ManualUpdatePanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        runpath="",
        notifier=notifier,
        parameter_configuration=[],
    )
    qtbot.addWidget(panel)

    ensemble_selector = panel.findChild(EnsembleSelector)
    assert ensemble_selector is not None

    index = ensemble_selector.findText(
        "ensemble_with_responses_only", Qt.MatchFlag.MatchContains
    )
    # The ensemble has data (responses), so it must be selectable
    assert index != -1
    ensemble_selector.setCurrentIndex(index)

    # Panel should not crash and active realizations should be empty
    realization_selector = panel.findChild(StringBox, "active_realizations_box")
    assert realization_selector is not None
    assert not realization_selector.text()
    assert not panel.isConfigurationValid()


def test_that_empty_experiment_name_field_defaults_to_manual_update_of_mock_ensemble0(
    qtbot: QtBot,
):
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    notifier._storage._setup_mocked_run(
        "mock_ensemble0",
        "mock_experiment0",
        [REALIZATION_FINISHED_SUCCESSFULLY, REALIZATION_FINISHED_SUCCESSFULLY],
    )
    panel = ManualUpdatePanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        runpath="",
        notifier=notifier,
        parameter_configuration=[],
    )
    qtbot.addWidget(panel)

    assert (
        panel.get_experiment_arguments().experiment_name
        == "Manual update of mock_ensemble0"
    )


def test_that_experiment_name_field_is_used_in_experiment_arguments(qtbot: QtBot):
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    notifier._storage._setup_mocked_run(
        "mock_ensemble0",
        "mock_experiment0",
        [REALIZATION_FINISHED_SUCCESSFULLY, REALIZATION_FINISHED_SUCCESSFULLY],
    )
    panel = ManualUpdatePanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        runpath="",
        notifier=notifier,
        parameter_configuration=[],
    )
    qtbot.addWidget(panel)

    panel._experiment_name_field.setText("my custom experiment")

    assert panel.get_experiment_arguments().experiment_name == "my custom experiment"


def test_that_strategy_summary_follows_selected_ensemble_and_is_hidden_for_enif(
    qtbot: QtBot,
):
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    for index, strategy in enumerate(
        [LocalizationType.ADAPTIVE, LocalizationType.DISTANCE]
    ):
        notifier._storage._setup_mocked_run(
            f"ensemble_{index}",
            f"experiment_{index}",
            [REALIZATION_FINISHED_SUCCESSFULLY],
            parameter_configuration={
                "parameter": GenKwConfig(
                    name="parameter",
                    distribution=RawSettings(),
                    update_strategy=strategy,
                )
            },
        )
    panel = ManualUpdatePanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        runpath="",
        notifier=notifier,
        parameter_configuration=[],
    )
    qtbot.addWidget(panel)

    summary = panel.findChild(UpdateStrategySummaryWidget)
    ensemble_selector = panel.findChild(EnsembleSelector)
    assert summary is not None
    assert ensemble_selector is not None

    ensemble_selector.setCurrentIndex(
        ensemble_selector.findText("experiment_1 : ensemble_1")
    )
    assert summary.rowCount() == 1
    summary_items = [summary.item(0, column) for column in range(3)]
    assert all(item is not None for item in summary_items)
    assert tuple(item.text() for item in summary_items if item is not None) == (
        "Distance",
        "GenKW",
        "1",
    )

    panel._update_method_dropdown.setCurrentText("EnIF Update (Experimental)")
    assert summary.isHidden()
    assert panel._update_strategy_label.isHidden()

    panel._update_method_dropdown.setCurrentText("ES Update")
    assert not summary.isHidden()
    assert not panel._update_strategy_label.isHidden()


def test_that_strategy_summary_shows_configured_parameters_when_no_ensemble_is_selected(
    qtbot: QtBot,
):
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    notifier._storage._setup_mocked_run(
        "ensemble",
        "experiment",
        [REALIZATION_FINISHED_SUCCESSFULLY],
        parameter_configuration={
            "stored": GenKwConfig(
                name="stored",
                distribution=RawSettings(),
                update_strategy=LocalizationType.DISTANCE,
            )
        },
    )
    panel = ManualUpdatePanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        runpath="",
        notifier=notifier,
        parameter_configuration=[
            GenKwConfig(
                name="configured",
                distribution=RawSettings(),
                update_strategy=LocalizationType.GLOBAL,
            )
        ],
    )
    qtbot.addWidget(panel)
    summary = panel.findChild(UpdateStrategySummaryWidget)
    ensemble_selector = panel.findChild(EnsembleSelector)
    assert summary is not None
    assert ensemble_selector is not None
    assert summary.item(0, 0).text() == "Distance"

    ensemble_selector.setCurrentIndex(-1)

    assert summary.rowCount() == 1
    assert summary.item(0, 0).text() == "Global"


def test_that_run_uses_parameter_configuration_of_selected_ensemble(qtbot: QtBot):
    configured_parameter = GenKwConfig(
        name="configured",
        distribution=RawSettings(),
        update_strategy=LocalizationType.GLOBAL,
    )
    stored_parameters = {
        f"ensemble_{index}": GenKwConfig(
            name="stored",
            distribution=RawSettings(),
            update_strategy=strategy,
        )
        for index, strategy in enumerate(
            [LocalizationType.ADAPTIVE, LocalizationType.DISTANCE]
        )
    }
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    for index, (ensemble_name, parameter) in enumerate(stored_parameters.items()):
        notifier._storage._setup_mocked_run(
            ensemble_name,
            f"experiment_{index}",
            [REALIZATION_FINISHED_SUCCESSFULLY],
            parameter_configuration={"stored": parameter},
        )
    panel = ManualUpdatePanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        runpath="",
        notifier=notifier,
        parameter_configuration=[configured_parameter],
    )
    qtbot.addWidget(panel)
    ensemble_selector = panel.findChild(EnsembleSelector)
    assert ensemble_selector is not None

    for index, (ensemble_name, parameter) in enumerate(stored_parameters.items()):
        ensemble_selector.setCurrentIndex(
            ensemble_selector.findText(f"experiment_{index} : {ensemble_name}")
        )
        assert panel.get_experiment_arguments().parameter_configuration == [parameter]


def test_that_prior_ensemble_strategies_are_locked_in_update_settings_dialog(
    qtbot: QtBot,
):
    configured_parameter = GenKwConfig(
        name="configured",
        distribution=RawSettings(),
        update_strategy=LocalizationType.GLOBAL,
    )
    prior_parameter = GenKwConfig(
        name="stored",
        distribution=RawSettings(),
        update_strategy=LocalizationType.DISTANCE,
    )
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    notifier._storage._setup_mocked_run(
        "ensemble",
        "experiment",
        [REALIZATION_FINISHED_SUCCESSFULLY] * 2,
        parameter_configuration={"stored": prior_parameter},
    )
    panel = ManualUpdatePanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        runpath="",
        notifier=notifier,
        parameter_configuration=[configured_parameter],
    )
    qtbot.addWidget(panel)
    assert panel.get_experiment_arguments().parameter_configuration == [prior_parameter]
    strategy_selectors_enabled: list[bool] = []

    def save_update_settings() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)
        strategy_selectors_enabled.extend(
            selector.isEnabled() for selector in dialog.findChildren(QComboBox)
        )
        click_dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Save)

    QTimer.singleShot(0, save_update_settings)
    edit_button = panel.findChild(QPushButton, "analysis_variables_popup_button")
    assert edit_button is not None
    assert edit_button.isEnabled()
    qtbot.mouseClick(edit_button, Qt.MouseButton.LeftButton)

    assert strategy_selectors_enabled
    assert not any(strategy_selectors_enabled)
    assert panel.get_experiment_arguments().parameter_configuration == [prior_parameter]
    assert prior_parameter.update_strategy == LocalizationType.DISTANCE
    assert configured_parameter.update_strategy == LocalizationType.GLOBAL
