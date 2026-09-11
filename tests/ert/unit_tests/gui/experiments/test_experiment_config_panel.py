from unittest.mock import Mock

import pytest

from ert.config.parameter_config import LocalizationType, ParameterConfig
from ert.gui.experiments.experiment_config_panel import has_updatable_parameters


@pytest.mark.parametrize(
    ("update_strategy", "expected"),
    [
        (None, False),
        (LocalizationType.GLOBAL, True),
    ],
)
def test_has_updatable_parameters(update_strategy, expected):
    param_mock: ParameterConfig = Mock(
        spec=ParameterConfig, update_strategy=update_strategy
    )
    parameter_configuration = [param_mock]
    assert has_updatable_parameters(parameter_configuration) is expected
