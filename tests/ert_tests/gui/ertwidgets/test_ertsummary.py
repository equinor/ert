from unittest.mock import MagicMock

import pytest
from ecl.util.util import StringList

from ert.gui.ertwidgets.models.ertsummary import ErtSummary


@pytest.fixture
def mock_ert(monkeypatch):
    ert_mock = MagicMock()

    string_list = StringList(["forward_model_1", "forward_model_2"])
    ert_mock.getModelConfig.return_value.getForwardModel.return_value.joblist.return_value = (  # noqa
        string_list
    )

    ert_mock.ensembleConfig.return_value.getKeylistFromVarType.return_value = [
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
