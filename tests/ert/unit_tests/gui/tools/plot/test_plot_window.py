from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pandas as pd
import pytest
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton
from pytestqt.qtbot import QtBot

from ert.config.distribution import RawSettings
from ert.config.gen_kw_config import DataSource, GenKwConfig
from ert.gui.plotting.ert_plots.gaussian_kde import plotGaussianKDE
from ert.gui.plotting.models import DataTypeSeparator
from ert.gui.plotting.plot_api import EnsembleObject, PlotApi, PlotApiKeyDefinition
from ert.gui.plotting.plot_window import (
    DISTRIBUTION,
    GAUSSIAN_KDE,
    HISTOGRAM,
    STATISTICS,
    PlotWindow,
    create_error_dialog,
    make_seismic_y_label,
)
from ert.gui.plotting.utils import PlotConfig, PlotContext
from ert.gui.plotting.widgets import DataTypeKeysWidget
from ert.gui.plotting.widgets.plot_widget import PlotWidget
from ert.services import ErtServerController

EVEREST_KEY_DEFS = [
    PlotApiKeyDefinition(
        "obj1",
        index_type="VALUE",
        metadata={"data_origin": "everest_objectives"},
        observations=False,
        dimensionality=2,
    ),
    PlotApiKeyDefinition(
        "ctrl1",
        index_type=None,
        metadata={"data_origin": "everest_parameters"},
        observations=False,
        dimensionality=1,
    ),
    PlotApiKeyDefinition(
        "con1",
        index_type="VALUE",
        metadata={"data_origin": "everest_constraints"},
        observations=False,
        dimensionality=2,
    ),
]

OBSERVATION_KEY_DEFS = [
    PlotApiKeyDefinition(
        "obs1",
        index_type="VALUE",
        metadata={"data_origin": "everest_observations"},
        observations=True,
        dimensionality=2,
    )
]


def test_pressing_copy_button_in_error_dialog(qtbot: QtBot):
    qd = create_error_dialog("hello", "world")
    qtbot.addWidget(qd)

    qtbot.mouseClick(
        qd.findChild(QPushButton, name="copy_button"), Qt.MouseButton.LeftButton
    )
    assert QApplication.clipboard().text() == "world"


@pytest.mark.slow
def test_that_no_data_message_is_displayed(
    qtbot: QtBot, tmp_path, monkeypatch, use_tmpdir
):
    mock_plot_api_cls = MagicMock(spec=PlotApi)
    mock_plot_api = MagicMock(spec=PlotApi)
    mock_plot_api_cls.return_value = mock_plot_api

    storage_version = "0.0"
    mock_plot_api.api_version = storage_version
    monkeypatch.setattr(
        "ert.gui.plotting.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr("ert.gui.plotting.plot_window.PlotApi", mock_plot_api_cls)

    with ErtServerController.init_service(project=tmp_path):
        pw = PlotWindow("", tmp_path, None)
        qtbot.addWidget(pw)
        pw.show()
        widget = pw._central_tab.currentWidget()

        assert isinstance(widget, PlotWidget)

        fig = widget._figure
        texts = fig.texts
        assert len(texts) == 1
        assert (
            texts[0].get_text()
            == "No data to visualize. Head over to 'Start experiment' to get started!"
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("key_defs", "expected"),
    [(OBSERVATION_KEY_DEFS, "Observations available"), ([], None)],
)
def test_that_legend_displays_correct_message(
    qtbot: QtBot, tmp_path, monkeypatch, use_tmpdir, key_defs, expected
):
    mock_plot_api_cls = MagicMock(spec=PlotApi)
    mock_plot_api = MagicMock(spec=PlotApi)
    mock_plot_api_cls.return_value = mock_plot_api

    storage_version = "0.0"
    mock_plot_api.api_version = storage_version
    monkeypatch.setattr(
        "ert.gui.plotting.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr("ert.gui.plotting.plot_window.PlotApi", mock_plot_api_cls)

    mock_plot_api.responses_api_key_defs = []
    mock_plot_api.parameters_api_key_defs = key_defs

    with ErtServerController.init_service(project=tmp_path):
        pw = PlotWindow("", tmp_path, None)
        qtbot.addWidget(pw)
        pw.show()

        legend_label = pw._data_type_keys_widget.findChild(
            QLabel, name="observation_legend_label"
        )
        if expected is None:
            assert not legend_label
        else:
            assert legend_label
            assert legend_label.text() == expected


@pytest.mark.slow
def test_that_observation_legend_is_not_displayed_when_its_everest(
    qtbot: QtBot, tmp_path, monkeypatch, use_tmpdir
):
    mock_plot_api_cls = MagicMock(spec=PlotApi)
    mock_plot_api = MagicMock(spec=PlotApi)
    mock_plot_api_cls.return_value = mock_plot_api

    storage_version = "0.0"
    mock_plot_api.api_version = storage_version
    monkeypatch.setattr(
        "ert.gui.plotting.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr("ert.gui.plotting.plot_window.PlotApi", mock_plot_api_cls)
    monkeypatch.setattr(
        "ert.gui.plotting.widgets.data_type_keys_widget.is_everest_application",
        lambda: True,
    )

    with ErtServerController.init_service(project=tmp_path):
        pw = PlotWindow("", tmp_path, None)
        qtbot.addWidget(pw)
        pw.show()

        legend_label = pw._data_type_keys_widget.findChild(
            QLabel, name="observation_legend_label"
        )
        assert not legend_label


@pytest.mark.slow
def test_warning_is_visible_on_incompatible_plot_api_version(
    qtbot: QtBot, tmp_path, monkeypatch, use_tmpdir
):
    mock_get_data = MagicMock()
    mock_get_data.return_value = "0.2"
    monkeypatch.setattr("ert.gui.plotting.plot_api.PlotApi.api_version", mock_get_data)

    with ErtServerController.init_service(project=tmp_path):
        pw = PlotWindow("", tmp_path, None)
        qtbot.addWidget(pw)
        pw.show()

        label = pw.findChild(QLabel, name="plot_api_warning_label")
        assert label
        assert label.isVisible()
        assert label.text().startswith("<b>Plot API version mismatch detected")


@pytest.mark.slow
def test_that_plotting_gen_kw_parameter_with_negative_values_hides_log_scale_checkbox(
    qtbot: QtBot, monkeypatch
):
    """Verify sidebar log-scale checkbox behavior for GEN_KW values.

    The checkbox is hidden for keys with negative values, and its checked state
    is preserved when switching back to a key that supports log scale.
    """
    mock_plot_api_cls = MagicMock(spec=PlotApi)
    mock_plot_api = MagicMock(spec=PlotApi)
    mock_plot_api_cls.return_value = mock_plot_api

    storage_version = "0.0"
    mock_plot_api.api_version = storage_version
    monkeypatch.setattr(
        "ert.gui.plotting.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr("ert.gui.plotting.plot_window.PlotApi", mock_plot_api_cls)

    plot_api_key_def_positive = PlotApiKeyDefinition(
        "gen_kw_a",
        index_type=None,
        metadata={"data_origin": "GEN_KW"},
        observations=False,
        dimensionality=1,
        parameter=GenKwConfig(
            name="gen_kw_a", distribution={"name": "uniform", "min": 0, "max": 1}
        ),
    )
    plot_api_key_def_negative = PlotApiKeyDefinition(
        "gen_kw_b",
        index_type=None,
        metadata={"data_origin": "GEN_KW"},
        observations=False,
        dimensionality=1,
        parameter=GenKwConfig(
            name="gen_kw_b", distribution={"name": "uniform", "min": -1, "max": 0}
        ),
    )

    mock_plot_api.responses_api_key_defs = []
    mock_plot_api.parameters_api_key_defs = [
        plot_api_key_def_positive,
        plot_api_key_def_negative,
    ]

    def mock_data_for_parameter(ensemble_id: str, parameter_key: str) -> pd.DataFrame:
        if parameter_key == "gen_kw_a":
            return pd.DataFrame({0: [0.1, 0.5, 0.9]})
        return pd.DataFrame({0: [-0.1, -0.5, -0.9]})

    mock_plot_api.data_for_parameter.side_effect = mock_data_for_parameter
    mock_plot_api.has_history_data.return_value = False

    mock_plot_api.get_all_ensembles.return_value = [
        EnsembleObject(
            "ensemble",
            "ensemble",
            False,
            "experiment",
            "2026-01-01T00:00:00",
        )
    ]

    plot_window = PlotWindow(config_file="", ens_path="", parent=None)
    qtbot.addWidget(plot_window)
    plot_window.show()

    plot_widget = plot_window._central_tab.currentWidget()
    assert plot_widget is not None

    log_checkbox = plot_window.findChild(QCheckBox, name="log_scale_checkbox")
    assert log_checkbox is not None
    assert log_checkbox.isVisible()
    assert not log_checkbox.isChecked()

    def get_x_axis_scale() -> str:
        return (
            cast(PlotWidget, plot_window._central_tab.currentWidget())
            ._figure.axes[0]
            .xaxis.get_scale()
        )

    assert get_x_axis_scale() == "linear"

    log_checkbox.setChecked(True)
    qtbot.waitUntil(lambda: get_x_axis_scale() == "log", timeout=5000)
    assert log_checkbox.isChecked()

    # Select the second key (gen_kw_b), which has negative values.
    data_type_keys_widget = plot_window._data_type_keys_widget.data_type_keys_widget
    data_type_keys_widget.setCurrentIndex(
        plot_window._data_type_keys_widget.filter_model.index(1, 0)
    )
    qtbot.waitUntil(lambda: not log_checkbox.isVisible(), timeout=5000)
    assert get_x_axis_scale() == "linear"

    # Switch back to the first key (gen_kw_a) with positive values.
    data_type_keys_widget.setCurrentIndex(
        plot_window._data_type_keys_widget.filter_model.index(0, 0)
    )
    qtbot.waitUntil(log_checkbox.isVisible, timeout=5000)
    assert log_checkbox.isChecked(), "Log scale checkbox should still be checked"
    assert get_x_axis_scale() == "log", "Plot scale should still be log"


@pytest.mark.parametrize(
    ("key", "history_data_available", "observations_available"),
    [
        pytest.param("summary", False, False, id="neither-available"),
        pytest.param("summary", True, False, id="history-available"),
        pytest.param("summary", False, True, id="observations-available"),
        pytest.param("summary", True, True, id="both-available"),
        pytest.param("summaryH", False, False, id="history-key"),
        pytest.param("summaryH", False, True, id="history-key-with-observations"),
        pytest.param("WOPRH:OP1", False, False, id="well-history-key"),
    ],
)
def test_that_history_and_observations_checkboxes_match_data_availability(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    history_data_available: bool,
    observations_available: bool,
) -> None:
    mock_plot_api_cls = MagicMock(spec=PlotApi)
    mock_plot_api = MagicMock(spec=PlotApi)
    mock_plot_api_cls.return_value = mock_plot_api

    storage_version = "0.0"
    mock_plot_api.api_version = storage_version
    monkeypatch.setattr(
        "ert.gui.plotting.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr("ert.gui.plotting.plot_window.PlotApi", mock_plot_api_cls)

    key_def = PlotApiKeyDefinition(
        key,
        index_type="TIME",
        metadata={"data_origin": "SUMMARY"},
        observations=observations_available,
        dimensionality=1,
        response=MagicMock(type="summary"),
    )
    mock_plot_api.responses_api_key_defs = [key_def]
    mock_plot_api.parameters_api_key_defs = []
    mock_plot_api.has_history_data.return_value = history_data_available
    mock_plot_api.get_all_ensembles.return_value = []

    plot_window = PlotWindow(config_file="", ens_path=Path(), parent=None)
    qtbot.addWidget(plot_window)
    plot_window.show()

    history_checkbox = plot_window.findChild(QCheckBox, name="history_checkbox")
    assert history_checkbox is not None
    is_history_key = key.endswith("H") or "H:" in key
    assert history_checkbox.isVisible() is history_data_available
    if is_history_key:
        mock_plot_api.has_history_data.assert_not_called()

    observations_checkbox = plot_window.findChild(
        QCheckBox, name="observations_checkbox"
    )
    assert observations_checkbox is not None
    assert observations_checkbox.isVisible() is observations_available


def test_that_history_and_observations_checkbox_state_update_when_switching_keys(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_plot_api_cls = MagicMock(spec=PlotApi)
    mock_plot_api = MagicMock(spec=PlotApi)
    mock_plot_api_cls.return_value = mock_plot_api

    storage_version = "0.0"
    mock_plot_api.api_version = storage_version
    monkeypatch.setattr(
        "ert.gui.plotting.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr("ert.gui.plotting.plot_window.PlotApi", mock_plot_api_cls)

    key_defs = [
        PlotApiKeyDefinition(
            "summary",
            index_type="TIME",
            metadata={"data_origin": "SUMMARY"},
            observations=True,
            dimensionality=1,
            response=MagicMock(type="summary"),
        ),
        PlotApiKeyDefinition(
            "summaryH",
            index_type="TIME",
            metadata={"data_origin": "SUMMARY"},
            observations=False,
            dimensionality=1,
            response=MagicMock(type="summary"),
        ),
    ]
    mock_plot_api.responses_api_key_defs = key_defs
    mock_plot_api.parameters_api_key_defs = []
    mock_plot_api.has_history_data.return_value = True
    mock_plot_api.get_all_ensembles.return_value = []

    plot_window = PlotWindow(config_file="", ens_path=Path(), parent=None)
    qtbot.addWidget(plot_window)
    plot_window.show()

    history_checkbox = plot_window.findChild(QCheckBox, name="history_checkbox")
    observations_checkbox = plot_window.findChild(
        QCheckBox, name="observations_checkbox"
    )
    assert history_checkbox is not None
    assert observations_checkbox is not None
    data_type_keys_widget = plot_window._data_type_keys_widget.data_type_keys_widget
    filter_model = plot_window._data_type_keys_widget.filter_model

    assert history_checkbox.isVisible()
    assert observations_checkbox.isVisible()

    history_checkbox.setChecked(False)
    observations_checkbox.setChecked(False)

    mock_plot_api.has_history_data.reset_mock()
    data_type_keys_widget.setCurrentIndex(filter_model.index(1, 0))
    qtbot.waitUntil(lambda: not history_checkbox.isVisible(), timeout=5000)
    qtbot.waitUntil(lambda: not observations_checkbox.isVisible(), timeout=5000)
    mock_plot_api.has_history_data.assert_not_called()

    data_type_keys_widget.setCurrentIndex(filter_model.index(0, 0))
    qtbot.waitUntil(history_checkbox.isVisible, timeout=5000)
    qtbot.waitUntil(observations_checkbox.isVisible, timeout=5000)
    mock_plot_api.has_history_data.assert_called_once_with("summary")
    assert not history_checkbox.isChecked()
    assert not observations_checkbox.isChecked()


def test_that_general_option_checkboxes_change_rendered_plot(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_plot_api_cls = MagicMock(spec=PlotApi)
    mock_plot_api = MagicMock(spec=PlotApi)
    mock_plot_api_cls.return_value = mock_plot_api

    storage_version = "0.0"
    mock_plot_api.api_version = storage_version
    monkeypatch.setattr(
        "ert.gui.plotting.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr("ert.gui.plotting.plot_window.PlotApi", mock_plot_api_cls)

    observations_enabled_states: list[bool] = []
    original_set_observations_enabled = PlotConfig.set_observations_enabled

    def record_observations_enabled(plot_config: PlotConfig, enabled: bool) -> None:
        observations_enabled_states.append(enabled)
        original_set_observations_enabled(plot_config, enabled)

    monkeypatch.setattr(
        PlotConfig,
        "set_observations_enabled",
        record_observations_enabled,
    )

    key_def = PlotApiKeyDefinition(
        "summary",
        index_type="TIME",
        metadata={"data_origin": "SUMMARY"},
        observations=True,
        dimensionality=2,
        response=MagicMock(type="summary"),
    )
    ensemble = EnsembleObject(
        "ensemble",
        "ensemble",
        False,
        "experiment",
        "2026-01-01T00:00:00",
    )
    mock_plot_api.responses_api_key_defs = [key_def]
    mock_plot_api.parameters_api_key_defs = []
    mock_plot_api.data_for_response.return_value = pd.DataFrame(
        {
            pd.Timestamp("2023-01-01"): [1.0, 2.0],
            pd.Timestamp("2023-01-02"): [2.0, 3.0],
        },
        index=pd.Index([0, 1], name="Realization"),
    )
    mock_plot_api.has_history_data.return_value = True
    mock_plot_api.history_data.return_value = pd.DataFrame(
        {"history": [1.5, 2.5]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )
    mock_plot_api.observations_for_key.return_value = pd.DataFrame()
    mock_plot_api.get_all_ensembles.return_value = [ensemble]

    plot_window = PlotWindow(config_file="", ens_path=Path(), parent=None)
    qtbot.addWidget(plot_window)
    plot_window.show()

    statistics_index = next(
        index
        for index in range(plot_window._central_tab.count())
        if plot_window._central_tab.tabText(index) == STATISTICS
    )
    plot_window._central_tab.setCurrentIndex(statistics_index)
    plot_widget = cast(PlotWidget, plot_window._central_tab.currentWidget())
    qtbot.waitUntil(lambda: len(plot_widget._figure.axes) == 1, timeout=5000)

    def current_axes():
        return plot_widget._figure.axes[0]

    def current_legend_labels() -> set[str]:
        legend = current_axes().get_legend()
        return (
            set()
            if legend is None
            else {text.get_text() for text in legend.get_texts()}
        )

    legend_checkbox = plot_window.findChild(QCheckBox, name="legend_checkbox")
    grid_checkbox = plot_window.findChild(QCheckBox, name="grid_checkbox")
    history_checkbox = plot_window.findChild(QCheckBox, name="history_checkbox")
    observations_checkbox = plot_window.findChild(
        QCheckBox, name="observations_checkbox"
    )
    assert legend_checkbox is not None
    assert grid_checkbox is not None
    assert history_checkbox is not None
    assert observations_checkbox is not None
    assert history_checkbox.isVisible()
    assert current_axes().get_legend() is not None
    assert "History" in current_legend_labels()
    assert any(gridline.get_visible() for gridline in current_axes().get_xgridlines())

    for checkbox in (
        history_checkbox,
        observations_checkbox,
        legend_checkbox,
        grid_checkbox,
    ):
        checkbox.setChecked(False)
    qtbot.waitUntil(
        lambda: "History" not in current_legend_labels(),
        timeout=5000,
    )
    qtbot.waitUntil(lambda: observations_enabled_states[-1] is False, timeout=5000)
    qtbot.waitUntil(lambda: current_axes().get_legend() is None, timeout=5000)
    qtbot.waitUntil(
        lambda: (
            not any(
                gridline.get_visible() for gridline in current_axes().get_xgridlines()
            )
        ),
        timeout=5000,
    )

    for checkbox in (
        history_checkbox,
        observations_checkbox,
        legend_checkbox,
        grid_checkbox,
    ):
        checkbox.setChecked(True)
    qtbot.waitUntil(
        lambda: "History" in current_legend_labels(),
        timeout=5000,
    )
    qtbot.waitUntil(lambda: current_axes().get_legend() is not None, timeout=5000)
    qtbot.waitUntil(
        lambda: any(
            gridline.get_visible() for gridline in current_axes().get_xgridlines()
        ),
        timeout=5000,
    )
    qtbot.waitUntil(lambda: observations_enabled_states[-1] is True, timeout=5000)

    general_options = plot_window._general_options.get_widget()
    std_dev_index = next(
        index
        for index in range(plot_window._central_tab.count())
        if plot_window._central_tab.tabText(index) == "Std dev"
    )
    plot_window._central_tab.setCurrentIndex(std_dev_index)
    qtbot.waitUntil(lambda: not general_options.isVisible(), timeout=5000)

    plot_window._central_tab.setCurrentIndex(statistics_index)
    qtbot.waitUntil(general_options.isVisible, timeout=5000)


def test_that_log_scale_state_is_preserved_when_switching_plot_tabs(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_plot_api_cls = MagicMock(spec=PlotApi)
    mock_plot_api = MagicMock(spec=PlotApi)
    mock_plot_api_cls.return_value = mock_plot_api

    storage_version = "0.0"
    mock_plot_api.api_version = storage_version
    monkeypatch.setattr(
        "ert.gui.plotting.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr(
        "ert.gui.plotting.plot_window.PlotApi",
        mock_plot_api_cls,
    )

    key_def = PlotApiKeyDefinition(
        "gen_kw",
        index_type=None,
        metadata={"data_origin": "GEN_KW"},
        observations=False,
        dimensionality=1,
        parameter=GenKwConfig(
            name="gen_kw",
            distribution={"name": "uniform", "min": 0, "max": 1},
        ),
    )

    mock_plot_api.responses_api_key_defs = []
    mock_plot_api.parameters_api_key_defs = [key_def]
    mock_plot_api.data_for_parameter.return_value = pd.DataFrame({0: [0.1, 0.5, 0.9]})
    mock_plot_api.has_history_data.return_value = False
    mock_plot_api.get_all_ensembles.return_value = [
        EnsembleObject(
            "ensemble",
            "ensemble",
            False,
            "experiment",
            "2026-01-01T00:00:00",
        )
    ]

    plot_window = PlotWindow(config_file="", ens_path=Path(), parent=None)
    qtbot.addWidget(plot_window)
    plot_window.show()

    tab_index_by_name = {
        plot_window._central_tab.tabText(index): index
        for index in range(plot_window._central_tab.count())
    }

    histogram_index = tab_index_by_name[HISTOGRAM]
    assert plot_window._central_tab.isTabEnabled(histogram_index)

    log_scale_tabs = {HISTOGRAM, DISTRIBUTION, GAUSSIAN_KDE}
    non_log_scale_index = next(
        index
        for index in range(plot_window._central_tab.count())
        if plot_window._central_tab.isTabEnabled(index)
        and plot_window._central_tab.tabText(index) not in log_scale_tabs
    )

    log_checkbox = plot_window.findChild(
        QCheckBox,
        name="log_scale_checkbox",
    )
    assert log_checkbox is not None

    plot_window._central_tab.setCurrentIndex(histogram_index)
    qtbot.waitUntil(log_checkbox.isVisible, timeout=5000)

    log_checkbox.setChecked(True)
    assert log_checkbox.isChecked()

    plot_window._central_tab.setCurrentIndex(non_log_scale_index)
    qtbot.waitUntil(lambda: not log_checkbox.isVisible(), timeout=5000)

    plot_window._central_tab.setCurrentIndex(histogram_index)
    qtbot.waitUntil(log_checkbox.isVisible, timeout=5000)

    assert log_checkbox.isChecked()


@pytest.mark.parametrize("tab_name", [HISTOGRAM, DISTRIBUTION, GAUSSIAN_KDE])
@pytest.mark.parametrize(
    ("values", "expected_visible"),
    [
        pytest.param([-0.1, -0.5, -0.9], False, id="negative-values"),
        pytest.param([0.1, 0.1, 0.1], False, id="constant-values"),
        pytest.param([0.1, 0.5, 0.9], True, id="positive-values"),
    ],
)
def test_that_density_tabs_show_log_scale_only_for_valid_gen_kw_values(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tab_name: str,
    values: list[float],
    expected_visible: bool,
) -> None:
    mock_plot_api_cls = MagicMock(spec=PlotApi)
    mock_plot_api = MagicMock(spec=PlotApi)
    mock_plot_api_cls.return_value = mock_plot_api

    storage_version = "0.0"
    mock_plot_api.api_version = storage_version
    monkeypatch.setattr(
        "ert.gui.plotting.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr("ert.gui.plotting.plot_window.PlotApi", mock_plot_api_cls)

    key_def = PlotApiKeyDefinition(
        "gen_kw",
        index_type=None,
        metadata={"data_origin": "GEN_KW"},
        observations=False,
        dimensionality=1,
        parameter=GenKwConfig(
            name="gen_kw",
            distribution={"name": "uniform", "min": 0.0, "max": 1.0},
        ),
    )

    mock_plot_api.responses_api_key_defs = []
    mock_plot_api.parameters_api_key_defs = [key_def]
    mock_plot_api.data_for_parameter.return_value = pd.DataFrame({0: values})
    mock_plot_api.has_history_data.return_value = False
    mock_plot_api.get_all_ensembles.return_value = [
        EnsembleObject(
            "ensemble",
            "ensemble",
            False,
            "experiment",
            "2026-01-01T00:00:00",
        )
    ]

    plot_window = PlotWindow(config_file="", ens_path=Path(), parent=None)
    qtbot.addWidget(plot_window)
    plot_window.show()

    tab_index = next(
        index
        for index in range(plot_window._central_tab.count())
        if plot_window._central_tab.tabText(index) == tab_name
    )
    plot_window._central_tab.setCurrentIndex(tab_index)

    log_checkbox = plot_window.findChild(QCheckBox, name="log_scale_checkbox")
    assert log_checkbox is not None
    qtbot.waitUntil(
        lambda: log_checkbox.isVisible() is expected_visible,
        timeout=5000,
    )


@pytest.mark.slow
def test_that_plot_window_ignores_negative_check_for_non_numeric_columns(
    qtbot: QtBot, monkeypatch
):
    """Regression test: gen_kw data may include non-numeric columns if they are
    from design matrix. Those values should be ignored when checking for
    negative values to determine whether log scale is possible.
    """

    mock_plot_api_cls = MagicMock(spec=PlotApi)
    mock_plot_api = MagicMock(spec=PlotApi)
    mock_plot_api_cls.return_value = mock_plot_api

    storage_version = "0.0"
    mock_plot_api.api_version = storage_version
    monkeypatch.setattr(
        "ert.gui.plotting.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr("ert.gui.plotting.plot_window.PlotApi", mock_plot_api_cls)

    plot_api_key_def = PlotApiKeyDefinition(
        "animal_type",
        index_type=None,
        metadata={"data_origin": "GEN_KW"},
        observations=False,
        dimensionality=1,
        parameter=GenKwConfig(
            name="animal_type",
            distribution=RawSettings(),
            group="DESIGN_MATRIX",
            input_source=DataSource.DESIGN_MATRIX,
        ),
    )

    mock_plot_api.responses_api_key_defs = []
    mock_plot_api.parameters_api_key_defs = [plot_api_key_def]

    def mixed_dtype_data_for_parameter(
        ensemble_id: str, parameter_key: str
    ) -> pd.DataFrame:
        assert parameter_key == "animal_type"
        return pd.DataFrame(
            {
                0: ["cat", "dog", "fish"],
            }
        )

    mock_plot_api.data_for_parameter.side_effect = mixed_dtype_data_for_parameter
    mock_plot_api.has_history_data.return_value = False
    mock_plot_api.get_all_ensembles.return_value = [
        EnsembleObject(
            "ensemble",
            "ensemble",
            False,
            "experiment",
            "2026-01-01T00:00:00",
        )
    ]

    plot_window = PlotWindow(config_file="", ens_path=Path(), parent=None)
    qtbot.addWidget(plot_window)
    plot_window.show()

    # This is the call that previously crashed with TypeError.
    plot_window.updatePlot()


def test_that_gaussian_kde_plot_skips_categorical_data_without_raising():
    ensemble = EnsembleObject(
        "ensemble",
        "ensemble",
        False,
        "experiment",
        "2026-01-01T00:00:00",
    )
    categorical_df = pd.DataFrame({0: ["cat", "dog", "fish"], 1: [12, 12, 12]})

    fig = Figure()
    ctx = PlotContext(
        PlotConfig(),
        ensembles=[ensemble],
        ensembles_color_indexes=[0],
        key="animal_type",
        layer=None,
    )

    # Should not raise (categorical data is simply skipped).
    plotGaussianKDE(fig, ctx, {ensemble: categorical_df}, _observation_data=None)


def test_that_separators_are_included_in_everest(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ert.gui.plotting.widgets.data_type_keys_widget.is_everest_application",
        lambda: True,
    )

    widget = DataTypeKeysWidget(EVEREST_KEY_DEFS)
    qtbot.addWidget(widget)
    model = widget.model
    assert any(isinstance(item, DataTypeSeparator) for item in model._keys)


def test_that_datatype_separators_are_not_selectable(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ert.gui.plotting.widgets.data_type_keys_widget.is_everest_application",
        lambda: True,
    )
    widget = DataTypeKeysWidget(EVEREST_KEY_DEFS)
    qtbot.addWidget(widget)
    model = widget.model
    for i, item in enumerate(model._keys):
        if isinstance(item, DataTypeSeparator):
            idx = model.index(i, 0)
            flags = model.flags(idx)
            assert not (flags & Qt.ItemFlag.ItemIsSelectable)


def test_that_datatype_separators_are_never_set_as_default(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ert.gui.plotting.widgets.data_type_keys_widget.is_everest_application",
        lambda: True,
    )
    widget = DataTypeKeysWidget(EVEREST_KEY_DEFS)
    qtbot.addWidget(widget)
    widget.selectDefault()
    selected = widget.getSelectedItem()
    assert selected is not None
    assert isinstance(selected, PlotApiKeyDefinition)


@pytest.mark.parametrize(
    ("key", "expected_y_label"),
    [
        pytest.param(
            "topvolantis--relai_full_rms_depth--20180701_20180101",
            "Rms Relai",
            id="parsable key",
        ),
        pytest.param("cat1_cat2_cat3", "Value", id="unexpected key"),
    ],
)
def test_that_seismic_y_label_is_created(key, expected_y_label):
    label = make_seismic_y_label(key)
    assert label == expected_y_label
