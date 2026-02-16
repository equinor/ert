from typing import cast
from unittest.mock import MagicMock

import pandas as pd
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton
from pytestqt.qtbot import QtBot

from ert.config.distribution import RawSettings
from ert.config.gen_kw_config import DataSource, GenKwConfig
from ert.gui.tools.plot.plot_api import EnsembleObject, PlotApi, PlotApiKeyDefinition
from ert.gui.tools.plot.plot_widget import PlotWidget
from ert.gui.tools.plot.plot_window import PlotWindow, create_error_dialog
from ert.services import ErtServerController


def test_pressing_copy_button_in_error_dialog(qtbot: QtBot):
    qd = create_error_dialog("hello", "world")
    qtbot.addWidget(qd)

    qtbot.mouseClick(
        qd.findChild(QPushButton, name="copy_button"), Qt.MouseButton.LeftButton
    )
    assert QApplication.clipboard().text() == "world"


@pytest.mark.integration_test
def test_warning_is_visible_on_incompatible_plot_api_version(
    qtbot: QtBot, tmp_path, monkeypatch, use_tmpdir
):
    mock_get_data = MagicMock()
    mock_get_data.return_value = "0.2"
    monkeypatch.setattr(
        "ert.gui.tools.plot.plot_api.PlotApi.api_version", mock_get_data
    )

    with ErtServerController.init_service(project=tmp_path):
        pw = PlotWindow("", tmp_path, None)
        qtbot.addWidget(pw)
        pw.show()

        label = pw.findChild(QLabel, name="plot_api_warning_label")
        assert label
        assert label.isVisible()
        assert label.text().startswith("<b>Plot API version mismatch detected")


@pytest.mark.integration_test
def test_that_plotting_gen_kw_parameter_with_negative_values_hides_log_scale_checkbox(
    qtbot: QtBot, monkeypatch
):
    """This test verifies that the log scale checkbox is hidden when plotting a gen_kw
    with negative values. It also makes sure that the plotter remembers if log scale
    was used, and re-applies it when switching back."""
    mock_plot_api_cls = MagicMock(spec=PlotApi)
    mock_plot_api = MagicMock(spec=PlotApi)
    mock_plot_api_cls.return_value = mock_plot_api

    storage_version = "0.0"
    mock_plot_api.api_version = storage_version
    monkeypatch.setattr(
        "ert.gui.tools.plot.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr("ert.gui.tools.plot.plot_window.PlotApi", mock_plot_api_cls)

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

    log_checkbox = plot_widget.findChild(QCheckBox, name="log_scale_checkbox")
    assert log_checkbox.isVisible()
    assert not log_checkbox.isChecked()

    def get_x_axis_scale() -> str:
        return (
            cast(PlotWidget, plot_window._central_tab.currentWidget())
            ._figure.axes[0]
            .xaxis.get_scale()
        )

    assert get_x_axis_scale() == "linear"

    qtbot.mouseClick(log_checkbox, Qt.MouseButton.LeftButton)
    # Default selection is the first key (gen_kw_a), which has only positive values.
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


@pytest.mark.integration_test
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
        "ert.gui.tools.plot.plot_window.get_storage_api_version",
        lambda: storage_version,
    )
    monkeypatch.setattr("ert.gui.tools.plot.plot_window.PlotApi", mock_plot_api_cls)

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
                "animal_type": ["cat", "dog", "fish"],
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

    plot_window = PlotWindow(config_file="", ens_path="", parent=None)
    qtbot.addWidget(plot_window)
    plot_window.show()

    # This is the call that previously crashed with TypeError.
    plot_window.updatePlot()
