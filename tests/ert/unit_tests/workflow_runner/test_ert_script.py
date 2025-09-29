import sys

import pytest

from ert import ErtScript

from .workflow_common import WorkflowCommon


def test_failing_ert_script_provide_user_warning():
    class FailingScript(ErtScript):
        def run(self):
            raise UserWarning("Custom user warning")

    script = FailingScript()
    result = script.initializeAndRun([], [])
    assert script.hasFailed()
    assert result == "Custom user warning"


def test_initialize_and_run_converts_argument_types():
    class AddScript(ErtScript):
        def run(self, *arg):
            return arg[0] + arg[1]

    assert AddScript().initializeAndRun([int, int], ["5", "4"]) == 9

    with pytest.raises(ValueError, match="invalid literal for int"):
        AddScript().initializeAndRun([int, int], ["5", "4.6"])


def test_initialize_and_run_does_not_convert_none():
    class NoneScript(ErtScript):
        def run(self, arg):
            assert arg is None

    # Check if None is not converted to string "None"
    _ = NoneScript().initializeAndRun([str], [None])


@pytest.mark.usefixtures("use_tmpdir")
def test_ert_script_from_file():
    WorkflowCommon.createErtScriptsJob()
    subtract_script = ErtScript.loadScriptFromFile("subtract_script.py")()
    assert subtract_script.initializeAndRun([int, int], ["1", "2"]) == -1


@pytest.mark.usefixtures("use_tmpdir")
def test_ert_script_with_syntax_error_raises_value_error():
    with open("syntax_error_script.py", "w", encoding="utf-8") as f:
        f.write("from ert not_legal_syntax ErtScript\n")
    with pytest.raises(ValueError, match=r"ErtScript .*.py contains syntax error"):
        _ = ErtScript.loadScriptFromFile("syntax_error_script.py")


@pytest.mark.usefixtures("use_tmpdir")
def test_ert_script_with_import_error_raises_value_error():
    with open("import_error_script.py", "w", encoding="utf-8") as f:
        f.write("from ert import DoesNotExist\n")
    with pytest.raises(ValueError, match="cannot import name 'DoesNotExist'"):
        _ = ErtScript.loadScriptFromFile("import_error_script.py")


@pytest.mark.usefixtures("use_tmpdir")
def test_empty_ert_script_raises_value_error():
    with open("empty_script.py", "w", encoding="utf-8") as f:
        f.write("from ert import ErtScript\n")

    with pytest.raises(ValueError, match="does not contain an ErtScript"):
        _ = ErtScript.loadScriptFromFile("empty_script.py")


def test_that_exits_in_ert_script_is_trapped():
    class FailingScript(ErtScript):
        def run(self, *arg):
            sys.exit(-1)

    failing = FailingScript()
    failing.initializeAndRun([], [])
    assert failing.hasFailed()
