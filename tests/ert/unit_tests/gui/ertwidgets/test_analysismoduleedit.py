import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pytestqt.qtbot import QtBot

from ert.config import ESSettings
from ert.gui.ertwidgets.analysismoduleedit import AnalysisModuleEdit


@pytest.fixture
def es_settings() -> ESSettings:
    settings = ESSettings()
    settings.localization_correlation_threshold = 0.5
    settings.enkf_truncation = 0.2
    return settings


def _create_analysis_module_edit(
    qtbot: QtBot, es_settings: ESSettings, ensemble_size: int = 10
) -> AnalysisModuleEdit:
    widget = AnalysisModuleEdit(es_settings=es_settings, ensemble_size=ensemble_size)
    qtbot.addWidget(widget)
    return widget


def test_that_editing_spinners_writes_to_es_settings(
    qtbot: QtBot, es_settings: ESSettings
):
    panel = _create_analysis_module_edit(qtbot, es_settings).variables_panel

    panel.threshold_spinner.setValue(0.7)
    panel.truncation_spinner.setValue(0.3)

    assert es_settings.localization_correlation_threshold == pytest.approx(0.7)
    assert es_settings.enkf_truncation == pytest.approx(0.3)


def test_that_refresh_shows_changes_made_to_es_settings_elsewhere(
    qtbot: QtBot, es_settings: ESSettings
):
    widget = _create_analysis_module_edit(qtbot, es_settings)

    es_settings.localization_correlation_threshold = 0.8
    es_settings.enkf_truncation = 0.4
    widget.refresh()

    assert math.isclose(widget.variables_panel.threshold_spinner.value(), 0.8)
    assert math.isclose(widget.variables_panel.truncation_spinner.value(), 0.4)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    initial_ensemble_size=st.integers(min_value=1, max_value=10_000),
    new_ensemble_size=st.integers(min_value=1, max_value=10_000),
)
def test_that_unset_correlation_threshold_shows_default_for_ensemble_size(
    qtbot: QtBot, initial_ensemble_size: int, new_ensemble_size: int
):
    settings = ESSettings()
    widget = _create_analysis_module_edit(
        qtbot, settings, ensemble_size=initial_ensemble_size
    )
    spinner = widget.variables_panel.threshold_spinner

    assert spinner.value() == pytest.approx(
        ESSettings().correlation_threshold(initial_ensemble_size), abs=1e-6
    )

    widget.set_ensemble_size(new_ensemble_size)

    assert spinner.value() == pytest.approx(
        ESSettings().correlation_threshold(new_ensemble_size), abs=1e-6
    )
    assert settings.localization_correlation_threshold is None
