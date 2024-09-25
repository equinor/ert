import shutil

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton, QTextEdit

from ert.config import ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.tools.manage_experiments import ManageExperimentsPanel
from ert.gui.tools.manage_experiments.storage_info_widget import (
    _EnsembleWidget,
    _EnsembleWidgetTabs,
    _ExperimentWidget,
    _WidgetType,
)
from ert.gui.tools.manage_experiments.storage_widget import StorageWidget
from ert.storage import Storage, open_storage
from ert.storage.realization_storage_state import RealizationStorageState
from tests.ert.ui_tests.cli.analysis.test_adaptive_localization import (
    run_cli_ES_with_case,
)


@pytest.mark.usefixtures("copy_poly_case")
def test_init_prior(qtbot, storage):
    config = ErtConfig.from_file("poly.ert")
    config.random_seed = 1234
    notifier = ErtNotifier(config.config_path)
    notifier.set_storage(storage)
    ensemble = storage.create_experiment(
        parameters=config.ensemble_config.parameter_configuration,
        responses=config.ensemble_config.response_configuration,
        name="my-experiment",
    ).create_ensemble(
        ensemble_size=config.model_config.num_realizations,
        name="prior",
    )
    notifier.set_current_ensemble(ensemble)
    assert (
        ensemble.get_ensemble_state()
        == [RealizationStorageState.UNDEFINED] * config.model_config.num_realizations
    )

    tool = ManageExperimentsPanel(
        config, notifier, config.model_config.num_realizations
    )
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.MouseButton.LeftButton,
    )
    assert (
        ensemble.get_ensemble_state()
        == [RealizationStorageState.INITIALIZED] * config.model_config.num_realizations
    )
    assert ensemble.load_parameters("COEFFS")[
        "transformed_values"
    ].mean() == pytest.approx(1.41487404)


@pytest.mark.usefixtures("copy_poly_case")
def test_that_init_updates_the_info_tab(qtbot, storage):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier(config.config_path)
    notifier.set_storage(storage)

    ensemble = storage.create_experiment(
        parameters=config.ensemble_config.parameter_configuration,
        responses=config.ensemble_config.response_configuration,
        observations=config.observations,
        name="my-experiment",
    ).create_ensemble(
        ensemble_size=config.model_config.num_realizations, name="default"
    )
    notifier.set_current_ensemble(ensemble)

    tool = ManageExperimentsPanel(
        config, notifier, config.model_config.num_realizations
    )

    html_edit = tool.findChild(QTextEdit, name="ensemble_state_text")
    assert not html_edit.toPlainText()

    # select the created ensemble
    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(
        0, 0, storage_widget._tree_view.model().index(0, 0)
    )
    storage_widget._tree_view.setCurrentIndex(model_index)

    # select the correct tab
    ensemble_widget = tool.findChild(_EnsembleWidget)
    ensemble_widget._currentTabChanged(1)

    assert "UNDEFINED" in html_edit.toPlainText()
    assert not "RealizationStorageState.UNDEFINED" in html_edit.toPlainText()

    # Change to the "initialize from scratch" tab
    tool.setCurrentIndex(1)
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.MouseButton.LeftButton,
    )

    # Change back to first tab
    tool.setCurrentIndex(0)
    ensemble_widget._currentTabChanged(1)
    assert "INITIALIZED" in html_edit.toPlainText()
    assert not "RealizationStorageState.INITIALIZED" in html_edit.toPlainText()

    # select the observation
    storage_info_widget = tool._storage_info_widget
    storage_info_widget._ensemble_widget._tab_widget.setCurrentIndex(
        _EnsembleWidgetTabs.OBSERVATIONS_TAB
    )
    observation_tree = storage_info_widget._ensemble_widget._observations_tree_widget
    model_index = observation_tree.model().index(
        0, 0, observation_tree.model().index(0, 0)
    )
    observation_tree.setCurrentIndex(model_index)
    assert (
        storage_info_widget._ensemble_widget._figure.axes[0].title.get_text()
        == "POLY_OBS"
    )


def test_experiment_view(
    qtbot, snake_oil_case_storage: ErtConfig, snake_oil_storage: Storage
):
    config = snake_oil_case_storage
    storage = snake_oil_storage

    notifier = ErtNotifier(config.config_path)
    notifier.set_storage(storage)

    tool = ManageExperimentsPanel(
        config, notifier, config.model_config.num_realizations
    )

    # select the experiment
    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(0, 0)
    storage_widget._tree_view.setCurrentIndex(model_index)
    assert (
        tool._storage_info_widget._content_layout.currentIndex()
        == _WidgetType.EXPERIMENT_WIDGET
    )

    experiment_widget = tool._storage_info_widget._content_layout.currentWidget()
    assert isinstance(experiment_widget, _ExperimentWidget)
    assert experiment_widget._name_label.text()
    assert experiment_widget._uuid_label.text()
    assert experiment_widget._parameters_text_edit.toPlainText()
    assert experiment_widget._responses_text_edit.toPlainText()
    assert experiment_widget._observations_text_edit.toPlainText()


def test_ensemble_view(
    qtbot, snake_oil_case_storage: ErtConfig, snake_oil_storage: Storage
):
    config = snake_oil_case_storage
    storage = snake_oil_storage

    notifier = ErtNotifier(config.config_path)
    notifier.set_storage(storage)

    tool = ManageExperimentsPanel(
        config, notifier, config.model_config.num_realizations
    )

    # select the ensemble
    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(
        0, 0, storage_widget._tree_view.model().index(0, 0)
    )
    storage_widget._tree_view.setCurrentIndex(model_index)
    assert (
        tool._storage_info_widget._content_layout.currentIndex()
        == _WidgetType.ENSEMBLE_WIDGET
    )

    ensemble_widget = tool._storage_info_widget._content_layout.currentWidget()
    assert isinstance(ensemble_widget, _EnsembleWidget)
    assert ensemble_widget._name_label.text()
    assert ensemble_widget._uuid_label.text()
    assert not ensemble_widget._state_text_edit.toPlainText()

    ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.STATE_TAB)
    assert ensemble_widget._state_text_edit.toPlainText()

    ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.OBSERVATIONS_TAB)
    ensemble_widget._observations_tree_widget.expandAll()
    assert ensemble_widget._observations_tree_widget.topLevelItemCount() == 3
    assert ensemble_widget._observations_tree_widget.topLevelItem(0).childCount() == 200
    assert ensemble_widget._observations_tree_widget.topLevelItem(1).childCount() == 4
    assert ensemble_widget._observations_tree_widget.topLevelItem(2).childCount() == 6

    # simulate clicking some different entries in observation list
    ensemble_widget._observations_tree_widget.currentItemChanged.emit(
        ensemble_widget._observations_tree_widget.topLevelItem(0).child(10), None
    )
    assert ensemble_widget._figure.get_axes()[0].get_title() == "FOPR"

    ensemble_widget._observations_tree_widget.currentItemChanged.emit(
        ensemble_widget._observations_tree_widget.topLevelItem(1).child(2), None
    )
    assert ensemble_widget._figure.get_axes()[0].get_title() == "WPR_DIFF_1"

    ensemble_widget._observations_tree_widget.currentItemChanged.emit(
        ensemble_widget._observations_tree_widget.topLevelItem(2).child(3), None
    )
    assert ensemble_widget._figure.get_axes()[0].get_title() == "WOPR_OP1_108"


@pytest.mark.usefixtures("copy_poly_case")
def test_ensemble_observations_view(qtbot):
    with open("observations", "w", encoding="utf-8") as file:
        file.write(
            """GENERAL_OBSERVATION POLY_OBS {
        DATA       = POLY_RES;
        INDEX_LIST = 0,1,2,3,4;
        OBS_FILE   = poly_obs_data.txt;
    };
    GENERAL_OBSERVATION POLY_OBS1_1 {
        DATA       = POLY_RES1;
        INDEX_LIST = 0,1,2,3,4;
        OBS_FILE   = poly_obs_data1.txt;
    };
    GENERAL_OBSERVATION POLY_OBS1_2 {
        DATA       = POLY_RES2;
        INDEX_LIST = 0,1,2,3,4;
        OBS_FILE   = poly_obs_data2.txt;
    };
    """
        )

    with open("poly_eval.py", "w", encoding="utf-8") as file:
        file.write(
            """#!/usr/bin/env python3
import json


def _load_coeffs(filename):
    with open(filename, encoding="utf-8") as f:
        return json.load(f)["COEFFS"]


def _evaluate(coeffs, x):
    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]


if __name__ == "__main__":
    coeffs = _load_coeffs("parameters.json")
    output = [_evaluate(coeffs, x) for x in range(10)]
    with open("poly.out", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, output)))

    with open("poly.out1", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, [x*2 for x in output])))

    with open("poly.out2", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, [x*3 for x in output])))
"""
        )

    shutil.copy("poly_obs_data.txt", "poly_obs_data1.txt")
    shutil.copy("poly_obs_data.txt", "poly_obs_data2.txt")

    with open("poly_localization_0.ert", "w", encoding="utf-8") as f:
        f.write(
            """
        QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING 50

RUNPATH poly_out/realization-<IENS>/iter-<ITER>

OBS_CONFIG observations
REALIZATION_MEMORY 50mb

NUM_REALIZATIONS 100
MIN_REALIZATIONS 1

GEN_KW COEFFS coeff_priors
GEN_DATA POLY_RES RESULT_FILE:poly.out
GEN_DATA POLY_RES1 RESULT_FILE:poly.out1
GEN_DATA POLY_RES2 RESULT_FILE:poly.out2

INSTALL_JOB poly_eval POLY_EVAL
FORWARD_MODEL poly_eval

ANALYSIS_SET_VAR STD_ENKF LOCALIZATION True
ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD 0.0

ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE *
ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE POLY_OBS1_*
"""
        )

    prior_ens, _ = run_cli_ES_with_case("poly_localization_0.ert")
    config = ErtConfig.from_file("poly_localization_0.ert")

    notifier = ErtNotifier(config.config_path)
    with open_storage(config.ens_path, mode="w") as storage:
        notifier.set_storage(storage)

        tool = ManageExperimentsPanel(
            config, notifier, config.model_config.num_realizations
        )

        assert prior_ens.name

        # select the ensemble
        storage_widget = tool.findChild(StorageWidget)
        storage_widget._tree_view.expandAll()
        model_index = storage_widget._tree_view.model().index(
            0, 0, storage_widget._tree_view.model().index(0, 0)
        )
        storage_widget._tree_view.setCurrentIndex(model_index)
        assert (
            tool._storage_info_widget._content_layout.currentIndex()
            == _WidgetType.ENSEMBLE_WIDGET
        )

        ensemble_widget = tool._storage_info_widget._content_layout.currentWidget()
        assert isinstance(ensemble_widget, _EnsembleWidget)
        assert ensemble_widget._name_label.text()
        assert ensemble_widget._uuid_label.text()
        assert not ensemble_widget._state_text_edit.toPlainText()

        ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.STATE_TAB)
        assert ensemble_widget._state_text_edit.toPlainText()

        ensemble_widget._tab_widget.setCurrentIndex(
            _EnsembleWidgetTabs.OBSERVATIONS_TAB
        )

        # Check that a scaled observation is plotted
        assert any(
            l
            for l in ensemble_widget._figure.get_axes()[0].get_lines()
            if "Scaled observation" in l.get_xdata()
        )


def test_realization_view(
    qtbot, snake_oil_case_storage: ErtConfig, snake_oil_storage: Storage
):
    config = snake_oil_case_storage
    storage = snake_oil_storage

    notifier = ErtNotifier(config.config_path)
    notifier.set_storage(storage)

    tool = ManageExperimentsPanel(
        config, notifier, config.model_config.num_realizations
    )

    # select the realization
    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(
        0,
        0,
        storage_widget._tree_view.model().index(
            0, 0, storage_widget._tree_view.model().index(0, 0)
        ),
    )
    storage_widget._tree_view.setCurrentIndex(model_index)
    assert (
        tool._storage_info_widget._content_layout.currentIndex()
        == _WidgetType.REALIZATION_WIDGET
    )

    realization_widget = tool._storage_info_widget._content_layout.currentWidget()

    assert realization_widget._state_label.text() == "Realization state: HAS_DATA"
    assert {"gen_data - HAS_DATA", "summary - HAS_DATA"}.issubset(
        set(realization_widget._response_text_edit.toPlainText().splitlines())
    )
    assert (
        realization_widget._parameter_text_edit.toPlainText()
        == "\nSNAKE_OIL_PARAM - INITIALIZED\n"
    )
