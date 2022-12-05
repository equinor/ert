from unittest.mock import MagicMock

import pytest

from ert.shared.hook_implementations.workflows.disable_parameters import (
    DisableParametersUpdate,
)
from ert.shared.plugins import ErtPluginManager


@pytest.mark.parametrize(
    "input_string, expected", [("a", ["b", "c"]), ("a,b", ["c"]), ("a, b", ["c"])]
)
def test_parse_comma_list(tmpdir, monkeypatch, input_string, expected):

    ert_mock = MagicMock()
    ert_mock._observation_keys = ["OBSERVATION"]
    ert_mock._parameter_keys = ["a", "b", "c"]

    DisableParametersUpdate(ert_mock).run(input_string)
    assert ert_mock.update_configuration[0]["parameters"] == expected


def test_disable_parameters_is_loaded():
    pm = ErtPluginManager()
    assert "DISABLE_PARAMETERS" in pm.get_installable_workflow_jobs()
