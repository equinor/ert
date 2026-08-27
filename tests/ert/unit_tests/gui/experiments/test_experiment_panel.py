from unittest.mock import Mock, patch

from tests.ert.ui_tests.gui.conftest import ExperimentPanel


@patch.object(ExperimentPanel, "__init__", lambda *args, **kwargs: None)
def test_is_parameters_updatable():
    panel = ExperimentPanel.__new__(ExperimentPanel)
    panel.config = Mock()
    mock_param = Mock()

    mock_param.update_strategy = "some_strategy"
    panel.config.parameter_configurations_with_design_matrix = [mock_param]
    panel.config.observation_declarations = ["obs1"]
    assert panel._is_parameters_updatable() is True

    mock_param.update_strategy = None
    assert panel._is_parameters_updatable() is False

    mock_param.update_strategy = "some_strategy"
    panel.config.observation_declarations = []
    assert panel._is_parameters_updatable() is False


@patch.object(ExperimentPanel, "__init__", lambda *args, **kwargs: None)
def test_is_panel_enabled():
    panel = ExperimentPanel.__new__(ExperimentPanel)
    panel._parameters_updatable = True

    assert panel._is_panel_enabled(requires_updatable_parameters=True) is True
    assert panel._is_panel_enabled(requires_updatable_parameters=False) is True

    panel._parameters_updatable = False
    assert panel._is_panel_enabled(requires_updatable_parameters=True) is False
    assert panel._is_panel_enabled(requires_updatable_parameters=False) is True
