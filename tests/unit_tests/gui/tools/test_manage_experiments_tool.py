import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton, QTextEdit

from ert.config import ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.storage_info_widget import (
    _EnsembleWidget,
    _EnsembleWidgetTabs,
    _ExperimentWidget,
    _WidgetType,
)
from ert.gui.ertwidgets.storage_widget import StorageWidget
from ert.gui.tools.manage_experiments.ensemble_init_configuration import (
    EnsembleInitializationConfigurationPanel,
)
from ert.storage import Storage
from ert.storage.realization_storage_state import RealizationStorageState


@pytest.mark.usefixtures("copy_poly_case")
def test_init_prior(qtbot, storage):
    config = ErtConfig.from_file("poly.ert")
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
    ensemble.refresh_statemap()
    notifier.set_current_ensemble(ensemble)
    assert (
        ensemble.get_ensemble_state()
        == [RealizationStorageState.UNDEFINED] * config.model_config.num_realizations
    )

    tool = EnsembleInitializationConfigurationPanel(
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


@pytest.mark.usefixtures("copy_poly_case")
def test_that_init_updates_the_info_tab(qtbot, storage):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier(config.config_path)
    notifier.set_storage(storage)

    ensemble = storage.create_experiment(
        parameters=config.ensemble_config.parameter_configuration,
        responses=config.ensemble_config.response_configuration,
        observations=config.observations.datasets,
        name="my-experiment",
    ).create_ensemble(
        ensemble_size=config.model_config.num_realizations, name="default"
    )
    notifier.set_current_ensemble(ensemble)

    tool = EnsembleInitializationConfigurationPanel(
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

    tool = EnsembleInitializationConfigurationPanel(
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

    tool = EnsembleInitializationConfigurationPanel(
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


def test_realization_view(
    qtbot, snake_oil_case_storage: ErtConfig, snake_oil_storage: Storage
):
    config = snake_oil_case_storage
    storage = snake_oil_storage

    notifier = ErtNotifier(config.config_path)
    notifier.set_storage(storage)

    tool = EnsembleInitializationConfigurationPanel(
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
    # Fix these. They should not be UNDEFINED
    assert (
        realization_widget._response_text_edit.toPlainText()
        == "\nSNAKE_OIL_OPR_DIFF - HAS_DATA\nSNAKE_OIL_WPR_DIFF - HAS_DATA\nSNAKE_OIL_GPR_DIFF - HAS_DATA\nsummary - HAS_DATA\n"
    )
    assert (
        realization_widget._parameter_text_edit.toPlainText()
        == "\nSNAKE_OIL_PARAM - INITIALIZED\n"
    )
