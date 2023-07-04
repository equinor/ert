from unittest.mock import MagicMock

import pytest

from ert.config import Field, GenKwConfig, SurfaceConfig
from ert.gui.ertwidgets.models.ertsummary import ErtSummary


@pytest.fixture
def mock_ert(monkeypatch):
    ert_mock = MagicMock()

    ert_mock.resConfig.return_value.forward_model_job_name_list.return_value = [  # noqa
        "forward_model_1",
        "forward_model_2",
    ]

    ert_mock.ensembleConfig.return_value.parameter_configs = {
        "surface": MagicMock(spec=SurfaceConfig, ncol=10, nrow=7),
        "gen_kw": MagicMock(spec=GenKwConfig, transfer_functions=[1, 2, 3]),
        "field": MagicMock(spec=Field, nx=10, ny=5, nz=3),
    }

    yield ert_mock


def test_getForwardModels(mock_ert):
    expected_list = ["forward_model_1", "forward_model_2"]
    forward_model_list = ErtSummary(mock_ert).getForwardModels()
    assert forward_model_list == expected_list


def test_getParameters(mock_ert):
    expected_list = ["field (10, 5, 3)", "gen_kw (3)", "surface (10, 7)"]
    parameter_list, parameter_count = ErtSummary(mock_ert).getParameters()
    assert parameter_list == expected_list
    assert parameter_count == 223
