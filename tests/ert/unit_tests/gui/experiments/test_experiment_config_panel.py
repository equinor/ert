from unittest.mock import Mock

from ert.config.parameter_config import LocalizationType, ParameterConfig
from ert.gui.experiments.experiment_config_panel import has_updatable_parameters


def test_that_has_updatable_parameters_returns_false_when_absent():

    param_mock: ParameterConfig = Mock(spec=ParameterConfig, update_strategy=None)
    parameter_configuration = [param_mock]

    assert not has_updatable_parameters(parameter_configuration)


def test_that_has_updatable_parameters_returns_true_when_present():

    param_mock: ParameterConfig = Mock(
        spec=ParameterConfig, update_strategy=LocalizationType.GLOBAL
    )
    parameter_configuration = [param_mock]

    assert has_updatable_parameters(parameter_configuration)
