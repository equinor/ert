from unittest.mock import MagicMock

import pytest

from ert.gui.ertwidgets.models.ertsummary import ErtSummary


@pytest.fixture
def mock_ert(monkeypatch):
    ert_mock = MagicMock()

    ert_mock.resConfig.return_value.forward_model_job_name_list.return_value = [  # noqa
        "forward_model_1",
        "forward_model_2",
    ]

    ert_mock.ensembleConfig.return_value.parameters = [
        "param_1",
        "param_2",
    ]
    yield ert_mock


def test_getForwardModels(mock_ert):
    regular_list = ["forward_model_1", "forward_model_2"]

    forward_models = ErtSummary(mock_ert).getForwardModels()

    assert forward_models == regular_list


def test_getParameters(mock_ert):
    regular_list = ["param_1", "param_2"]

    parameters = ErtSummary(mock_ert).getParameters()

    assert parameters == regular_list
