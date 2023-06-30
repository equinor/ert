from unittest.mock import MagicMock

import pytest

from ert._c_wrappers.enkf import EnKFMain
from ert.config import ErtConfig
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

    DisableParametersUpdate(ert_mock, storage=None).run(input_string)
    assert ert_mock.update_configuration[0]["parameters"] == expected


def test_disable_parameters_is_loaded():
    pm = ErtPluginManager()
    assert "DISABLE_PARAMETERS" in pm.get_installable_workflow_jobs()


@pytest.mark.usefixtures("copy_poly_case")
def test_that_we_can_disable_a_parameter():
    with open("poly.ert", "a", encoding="utf-8") as fh:
        fh.writelines("GEN_KW DONT_UPDATE_KW template.txt kw.txt prior.txt")
    with open("template.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD <MY_KEYWORD>")
    with open("prior.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD NORMAL 0 1")
    ert = EnKFMain(ErtConfig.from_file("poly.ert"))

    # pylint: disable=no-member
    # (pylint is unable to read the members of update_step objects)

    parameters = [
        parameter.name
        for parameter in ert.update_configuration.update_steps[0].parameters
    ]
    assert "DONT_UPDATE_KW" in parameters
    DisableParametersUpdate(ert, storage=None).run("DONT_UPDATE_KW")

    parameters = [
        parameter.name
        for parameter in ert.update_configuration.update_steps[0].parameters
    ]
    assert "DONT_UPDATE_KW" not in parameters
