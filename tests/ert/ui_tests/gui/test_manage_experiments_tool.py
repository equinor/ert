import datetime
import shutil

import numpy as np
import polars as pl
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QTextEdit

from ert.config import ErtConfig, SummaryConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.tools.manage_experiments import ManageExperimentsPanel
from ert.gui.tools.manage_experiments.storage_info_widget import (
    _EnsembleWidget,
    _EnsembleWidgetTabs,
    _ExperimentWidget,
    _RealizationWidget,
    _WidgetType,
)
from ert.gui.tools.manage_experiments.storage_widget import StorageWidget
from ert.storage import RealizationStorageState, Storage, open_storage
from tests.ert.ui_tests.cli.analysis.test_adaptive_localization import (
    run_cli_ES_with_case,
)

from .conftest import add_experiment_in_manage_experiment_dialog


def test_design_matrix_in_manage_experiments_panel(
    copy_poly_case_with_design_matrix, qtbot, use_tmpdir
):
    num_realizations = 10
    a_values = list(range(num_realizations))
    design_dict = {
        "REAL": list(range(num_realizations)),
        "a": a_values,
    }
    default_list = [["b", 1], ["c", 2]]
    copy_poly_case_with_design_matrix(design_dict, default_list)
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier()
    notifier.set_storage(str(config.ens_path))
    assert config.ensemble_config.parameter_configuration == []
    assert config.analysis_config.design_matrix is not None

    with notifier.write_storage() as storage:
        storage.create_experiment(
            parameters=[config.analysis_config.design_matrix.parameter_configuration],
            responses=config.ensemble_config.response_configuration,
            name="my-experiment",
        ).create_ensemble(
            ensemble_size=config.runpath_config.num_realizations,
            name="my-design",
        )

    # Notifier storage is persistent, read-storage is not,
    # hence we get the ensemble from the read storage
    ensemble = notifier.storage.get_experiment_by_name(
        "my-experiment"
    ).get_ensemble_by_name("my-design")
    notifier.set_current_ensemble_id(ensemble.id)
    assert all(
        RealizationStorageState.UNDEFINED in s for s in ensemble.get_ensemble_state()
    )

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.MouseButton.LeftButton,
    )
    assert (
        RealizationStorageState.PARAMETERS_LOADED in s
        for s in ensemble.get_ensemble_state()
    )

    params = ensemble.load_parameters("DESIGN_MATRIX").drop("realization")
    np.testing.assert_array_equal(params["a"].to_list(), a_values)
    np.testing.assert_array_equal(params["b"].to_list(), np.ones(num_realizations))
    np.testing.assert_array_equal(params["c"].to_list(), 2 * np.ones(num_realizations))

    add_experiment_in_manage_experiment_dialog(
        qtbot, tool, experiment_name="my-experiment-2", ensemble_name="my-design-2"
    )

    experiments = list(notifier.storage.experiments)
    assert len(experiments) == 2

    # The write-storage writes the experiments,
    # and the read-storage refreshes itself.
    # There is no guarantee that the experiment UUIDs are in order-of-creation
    # hence, we do not assert the order here
    assert {e.name for e in experiments} == {"my-experiment", "my-experiment-2"}
    exp2 = notifier.storage.get_experiment_by_name("my-experiment-2")
    ensemble = exp2.get_ensemble_by_name("my-design-2")
    assert "DESIGN_MATRIX" in exp2.parameter_configuration
    assert {
        t.name
        for t in exp2.parameter_configuration[
            "DESIGN_MATRIX"
        ].transform_function_definitions
    } == {"a", "b", "c"}
    assert all(
        RealizationStorageState.UNDEFINED in s for s in ensemble.get_ensemble_state()
    )


@pytest.mark.usefixtures("copy_poly_case")
def test_init_prior(qtbot):
    config = ErtConfig.from_file("poly.ert")
    config.random_seed = 1234
    notifier = ErtNotifier()
    notifier.set_storage(config.ens_path)

    with notifier.write_storage() as storage:
        ensemble = storage.create_experiment(
            parameters=config.ensemble_config.parameter_configuration,
            responses=config.ensemble_config.response_configuration,
            name="my-experiment",
        ).create_ensemble(
            ensemble_size=config.runpath_config.num_realizations,
            name="prior",
        )

        assert all(
            RealizationStorageState.UNDEFINED in s
            for s in ensemble.get_ensemble_state()
        )
    notifier.set_current_ensemble_id(ensemble.id)

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.MouseButton.LeftButton,
    )
    assert (
        RealizationStorageState.PARAMETERS_LOADED in s
        for s in notifier.current_ensemble.get_ensemble_state()
    )
    assert notifier.current_ensemble.load_parameters_numpy(
        "COEFFS", np.arange(ensemble.ensemble_size)
    ).mean() == pytest.approx(0.0458710649708845)


@pytest.mark.usefixtures("copy_poly_case")
def test_that_init_updates_the_info_tab(qtbot):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier()
    notifier.set_storage(config.ens_path)

    with notifier.write_storage() as storage:
        ensemble = storage.create_experiment(
            parameters=config.ensemble_config.parameter_configuration,
            responses=config.ensemble_config.response_configuration,
            observations=config.observations,
            name="my-experiment",
        ).create_ensemble(
            ensemble_size=config.runpath_config.num_realizations, name="default"
        )
    notifier.set_current_ensemble_id(ensemble.id)

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
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
    assert "RealizationStorageState.UNDEFINED" not in html_edit.toPlainText()

    # Change to the "initialize from scratch" tab
    tool.setCurrentIndex(1)
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.MouseButton.LeftButton,
    )

    # Change back to first tab
    tool.setCurrentIndex(0)
    ensemble_widget._currentTabChanged(1)
    assert "PARAMETERS_LOADED" in html_edit.toPlainText()
    assert "RealizationStorageState.PARAMETERS_LOADED" not in html_edit.toPlainText()

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

    notifier = ErtNotifier()
    notifier.set_storage(str(storage.path))

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
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

    notifier = ErtNotifier()
    notifier.set_storage(str(storage.path))

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
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
QUEUE_OPTION LOCAL MAX_RUNNING 2

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

    prior_ens_id, _, _ = run_cli_ES_with_case(
        "poly_localization_0.ert", "test_experiment"
    )
    config = ErtConfig.from_file("poly_localization_0.ert")

    notifier = ErtNotifier()
    with open_storage(config.ens_path, mode="r") as storage:
        notifier.set_storage(str(storage.path))

        tool = ManageExperimentsPanel(
            config, notifier, config.runpath_config.num_realizations
        )

        assert storage.get_ensemble(prior_ens_id).name

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
            line
            for line in ensemble_widget._figure.get_axes()[0].get_lines()
            if "Scaled observation" in line.get_xdata()
        )


@pytest.mark.usefixtures("copy_poly_case")
def test_ensemble_observations_view_on_empty_ensemble(qtbot):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier()
    notifier.set_storage(config.ens_path)

    with notifier.write_storage() as storage:
        notifier.set_storage(str(storage.path))
        storage.create_experiment(
            responses=[SummaryConfig(keys=["*"])],
            observations={
                "summary": pl.DataFrame(
                    pl.DataFrame(
                        {
                            "response_key": ["FOPR"],
                            "observation_key": ["O4"],
                            "time": pl.Series(
                                [datetime.datetime(2000, 1, 1)],
                                dtype=pl.Datetime("ms"),
                            ),
                            "observations": pl.Series([10.2], dtype=pl.Float32),
                            "std": pl.Series([0.1], dtype=pl.Float32),
                        }
                    )
                ),
            },
        ).create_ensemble(
            name="test", ensemble_size=config.runpath_config.num_realizations
        )

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
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

    # Expect only one figure, the one for the observation
    assert len(ensemble_widget._figure.get_axes()) == 1


def test_realization_view(
    qtbot, snake_oil_case_storage: ErtConfig, snake_oil_storage: Storage
):
    config = snake_oil_case_storage
    storage = snake_oil_storage

    notifier = ErtNotifier()
    notifier.set_storage(str(storage.path))

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
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
    assert type(realization_widget) is _RealizationWidget

    assert (
        realization_widget._state_label.text()
        == "Realization state: PARAMETERS_LOADED, RESPONSES_LOADED"
    )
    assert {"gen_data - RESPONSES_LOADED", "summary - RESPONSES_LOADED"}.issubset(
        set(realization_widget._response_text_edit.toPlainText().splitlines())
    )
    assert (
        realization_widget._parameter_text_edit.toPlainText()
        == "\nSNAKE_OIL_PARAM - PARAMETERS_LOADED\n"
    )
