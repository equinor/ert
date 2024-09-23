import pytest

from ert import ErtScript

from .workflow_common import WorkflowCommon

# ruff: noqa: PLR6301


class AddScript(ErtScript):
    def run(self, *arg):
        return arg[0] + arg[1]


class NoneScript(ErtScript):
    def run(self, arg):
        assert arg is None


class FailingScript(ErtScript):
    def run(self):
        raise UserWarning("Custom user warning")


def test_failing_ert_script_provide_user_warning():
    script = FailingScript()
    result = script.initializeAndRun([], [])
    assert script.hasFailed()
    assert result == "Custom user warning"


def test_ert_script_add():
    script = AddScript()

    result = script.initializeAndRun([int, int], ["5", "4"])

    assert result == 9

    with pytest.raises(ValueError):
        result = script.initializeAndRun([int, int], ["5", "4.6"])


@pytest.mark.usefixtures("use_tmpdir")
def test_ert_script_from_file():
    WorkflowCommon.createErtScriptsJob()

    with open("syntax_error_script.py", "w", encoding="utf-8") as f:
        f.write("from ert not_legal_syntax ErtScript\n")

    with open("import_error_script.py", "w", encoding="utf-8") as f:
        f.write("from ert import DoesNotExist\n")

    with open("empty_script.py", "w", encoding="utf-8") as f:
        f.write("from ert import ErtScript\n")

    script_object = ErtScript.loadScriptFromFile("subtract_script.py")

    script = script_object()
    result = script.initializeAndRun([int, int], ["1", "2"])
    assert result == -1

    with pytest.raises(ValueError):
        _ = ErtScript.loadScriptFromFile("syntax_error_script.py")
    with pytest.raises(ValueError):
        _ = ErtScript.loadScriptFromFile("import_error_script.py")
    with pytest.raises(ValueError):
        _ = ErtScript.loadScriptFromFile("empty_script.py")


def test_none_ert_script():
    # Check if None is not converted to string "None"
    script = NoneScript()

    script.initializeAndRun([str], [None])
