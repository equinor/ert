from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QCheckBox
from pytestqt.qtbot import QtBot

from ert.config import EnsembleConfig
from ert.config.analysis_config import AnalysisConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import EnsembleSelector, StringBox
from ert.gui.simulation.multiple_data_assimilation_panel import (
    MultipleDataAssimilationPanel,
)
from ert.storage.local_experiment import ExperimentType

from .conftest import (
    REALIZATION_FINISHED_SUCCESSFULLY,
    REALIZATION_UNDEFINED,
    MockStorage,
)


def test_that_active_realizations_selector_validates_with_ensemble_size_from_config(
    qtbot: QtBot,
) -> None:
    """This is a test that makes sure the realization selector autofills and
    validates with the num_realizations from config/designmatrix, and the autofilled
    configuration is valid. It also makes sure the restart run button is disabled if
    there are no previous experiments/ensembles in storage"""
    active_realizations = [True, True, False, True, False, True, True]
    config_num_realizations = len(active_realizations)
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    panel = MultipleDataAssimilationPanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        parameter_configuration=EnsembleConfig().parameter_configuration,
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
    assert realization_selector.get_text == "0-1, 3, 5-6"
    assert not ensemble_selector.isEnabled()
    restart_checkbox = panel.findChild(QCheckBox, "restart_checkbox_esmda")
    assert not restart_checkbox.isEnabled()
    assert not restart_checkbox.isChecked()
    assert panel.isConfigurationValid()


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
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    notifier._storage._setup_mocked_run(
        "mock_ensemble",
        "mock_experiment",
        [
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_UNDEFINED,
            REALIZATION_UNDEFINED,
            REALIZATION_FINISHED_SUCCESSFULLY,
        ],
        experiment_type=ExperimentType.ES_MDA,
        iteration=0,
    )
    panel = MultipleDataAssimilationPanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        parameter_configuration=EnsembleConfig().parameter_configuration,
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


def test_multiple_data_assimilation_panel_sets_active_realizations_to_initial_active_realizations_when_restart_run_toggled(  # noqa: E501
    qtbot,
):
    active_realizations = [True, True, False, True, True]
    active_realizations_string = "0-1, 3-4"

    mock_notifier = MagicMock()
    mock_notifier.storage.get_unique_experiment_name.return_value = "foo"
    mda_panel = MultipleDataAssimilationPanel(
        analysis_config=MagicMock(),
        parameter_configuration=MagicMock(),
        run_path="",
        notifier=mock_notifier,
        active_realizations=active_realizations,
        config_num_realization=2,
    )
    assert mda_panel._active_realizations_field.text() == active_realizations_string
    mda_panel.restart_run_toggled()
    assert mda_panel._active_realizations_field.text() == active_realizations_string


@dataclass(frozen=True)
class EnsInfo:
    ensemble_id: str
    experiment_id: str
    parent: str | None
    experiment_type: ExperimentType
    iteration: int = 0

    @property
    def ensemble(self) -> str:
        return self.ensemble_id + "_ens"

    @property
    def experiment(self) -> str:
        return self.experiment_id + "_exp"


@pytest.mark.parametrize(
    ("extra_ensembles", "expected_ensembles"),
    [
        pytest.param(
            [
                EnsInfo("a", "mda", None, ExperimentType.ES_MDA, 0),
                EnsInfo("b", "mda", "a_ens", ExperimentType.ES_MDA, 1),
                EnsInfo("c", "mda", "b_ens", ExperimentType.ES_MDA, 2),
                EnsInfo("leaf", "mda", "c_ens", ExperimentType.ES_MDA, 3),
                EnsInfo("faux_child1", "e1", "c_ens", ExperimentType.MANUAL_UPDATE),
                EnsInfo("faux_child2", "e2", "leaf_ens", ExperimentType.MANUAL_UPDATE),
            ],
            [
                "mda_exp : a_ens",
                "mda_exp : b_ens",
                "mda_exp : c_ens",
            ],
            id="ES-MDA ensembles when not a leaf in the current ES_MDA",
        ),
        pytest.param(
            [
                EnsInfo(
                    "ensemble", "experiment", None, ExperimentType.ENSEMBLE_EXPERIMENT
                ),
                EnsInfo("a", "mda", None, ExperimentType.UNDEFINED, 0),
                EnsInfo("b", "mda", "a_ens", ExperimentType.UNDEFINED, 1),
            ],
            [
                "experiment_exp : ensemble_ens",
                "mda_exp : a_ens",
                "mda_exp : b_ens",
            ],
            id="both ensemble runs and old-storage ES_MDA non-leafs",
        ),
    ],
)
def test_that_restart_ensemble_select_contains_elements(
    qtbot: QtBot, extra_ensembles, expected_ensembles
) -> None:
    config_num_realizations = 5
    active_realizations = [True] * config_num_realizations
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    storage = notifier._storage
    ensembles_that_should_not_show_up = [
        EnsInfo("test", "test", None, ExperimentType.SINGLE_TEST_RUN),
        EnsInfo("manual", "manual", None, ExperimentType.MANUAL),
        EnsInfo("es0", "es", None, ExperimentType.ENSEMBLE_SMOOTHER, 0),
        EnsInfo("es1", "es", "es0_ens", ExperimentType.ENSEMBLE_SMOOTHER, 1),
    ]
    ensembles = ensembles_that_should_not_show_up + extra_ensembles
    for ensemble_info in ensembles:
        storage._setup_mocked_run(
            ensemble_info.ensemble,
            ensemble_info.experiment,
            [
                REALIZATION_FINISHED_SUCCESSFULLY,
            ],
            experiment_type=ensemble_info.experiment_type,
            iteration=ensemble_info.iteration,
        )
        if ensemble_info.parent is not None:
            keys = list(storage._ensembles.keys())
            storage._ensembles[keys[-1]]._index.prior_ensemble_id = next(
                k for k in keys if storage._ensembles[k].name == ensemble_info.parent
            )

    panel = MultipleDataAssimilationPanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        parameter_configuration=EnsembleConfig().parameter_configuration,
        run_path="",
        notifier=notifier,
        active_realizations=active_realizations,
        config_num_realization=config_num_realizations,
    )

    qtbot.addWidget(panel)
    ensemble_selector = panel.findChild(EnsembleSelector)
    assert ensemble_selector.count() == len(expected_ensembles)
    for i in range(ensemble_selector.count()):
        assert ensemble_selector.itemText(i) in expected_ensembles
