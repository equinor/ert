import inspect
from unittest.mock import MagicMock

import pytest

from ecl.util.util import StringList
from ert_gui.ertwidgets.models.ertsummary import ErtSummary


@pytest.fixture
def mock_ert_summary(monkeypatch, autouse=True):
    ERT_mock = MagicMock()
    test_module = inspect.getmodule(ErtSummary)
    monkeypatch.setattr(test_module, "ERT", ERT_mock)

    string_list = StringList(["forward_model_1", "forward_model_2"])
    ERT_mock.ert.getModelConfig.return_value.getForwardModel.return_value.joblist.return_value = (
        string_list
    )

    ERT_mock.ert.ensembleConfig.return_value.getKeylistFromVarType.return_value = [
        "param_1",
        "param_2",
    ]


@pytest.mark.usefixtures("mock_ert_summary")
def test_getForwardModels():
    regular_list = ["forward_model_1", "forward_model_2"]

    forward_models = ErtSummary().getForwardModels()

    assert forward_models == regular_list


@pytest.mark.usefixtures("mock_ert_summary")
def test_getParameters():
    regular_list = ["param_1", "param_2"]

    parameters = ErtSummary().getParameters()

    assert parameters == regular_list
