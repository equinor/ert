import sys
import threading
from pathlib import Path

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
    Path("syntax_error_script.py").write_text(
        "from ert not_legal_syntax ErtScript\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"ErtScript .*.py contains syntax error"):
        _ = ErtScript.loadScriptFromFile("syntax_error_script.py")


@pytest.mark.usefixtures("use_tmpdir")
def test_ert_script_with_import_error_raises_value_error():
    Path("import_error_script.py").write_text(
        "from ert import DoesNotExist\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cannot import name 'DoesNotExist'"):
        _ = ErtScript.loadScriptFromFile("import_error_script.py")


@pytest.mark.usefixtures("use_tmpdir")
def test_empty_ert_script_raises_value_error():
    Path("empty_script.py").write_text("from ert import ErtScript\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain an ErtScript"):
        _ = ErtScript.loadScriptFromFile("empty_script.py")


def test_that_exits_in_ert_script_is_trapped():
    class FailingScript(ErtScript):
        def run(self, *arg):
            sys.exit(-1)

    failing = FailingScript()
    failing.initializeAndRun([], [])
    assert failing.hasFailed()


def test_that_stdout_and_stderr_printed_by_an_ert_script_are_captured():
    class PrintingScript(ErtScript):
        def run(self):
            print("to stdout")
            print("to stderr", file=sys.stderr)

    script = PrintingScript()
    script.initializeAndRun([], [])

    assert script.stdoutdata == "to stdout\n"
    assert script.stderrdata == "to stderr\n"


def test_that_output_printed_before_an_ert_script_raises_is_captured():
    class PrintingAndFailingScript(ErtScript):
        def run(self):
            print("printed before failing")
            raise ValueError("boom")

    script = PrintingAndFailingScript()
    script.initializeAndRun([], [])

    assert script.hasFailed()
    assert script.stdoutdata == "printed before failing\n"


def test_that_the_stack_trace_of_a_failing_script_is_appended_to_captured_stderr():
    class PrintingAndFailingScript(ErtScript):
        def run(self):
            print("printed to stderr", file=sys.stderr)
            raise ValueError("boom")

    script = PrintingAndFailingScript()
    script.initializeAndRun([], [])

    assert script.stderrdata.startswith("printed to stderr\n")
    assert "ValueError: boom" in script.stderrdata


def test_that_output_captured_from_an_ert_script_is_still_written_to_stdout(capsys):
    class PrintingScript(ErtScript):
        def run(self):
            print("to stdout")
            print("to stderr", file=sys.stderr)

    PrintingScript().initializeAndRun([], [])

    captured = capsys.readouterr()
    assert captured.out == "to stdout\n"
    assert captured.err == "to stderr\n"


def _join(thread: threading.Thread) -> None:
    thread.join(timeout=10)
    assert not thread.is_alive(), (
        f"{thread.name} did not finish; it would leave sys.stdout captured"
    )


def test_that_output_written_by_another_thread_is_left_out_of_the_capture():
    job_may_finish = threading.Event()
    other_thread_has_printed = threading.Event()

    class SlowScript(ErtScript):
        def run(self):
            print("from the job")
            other_thread_has_printed.wait(timeout=10)
            job_may_finish.wait(timeout=10)

    def print_from_another_thread():
        print("from an unrelated thread")
        other_thread_has_printed.set()

    script = SlowScript()
    job = threading.Thread(target=script.initializeAndRun, args=([], []))
    job.start()
    other_thread = threading.Thread(target=print_from_another_thread)
    other_thread.start()
    _join(other_thread)
    assert other_thread_has_printed.is_set(), "the unrelated thread never printed"
    job_may_finish.set()
    _join(job)

    assert "from the job" in script.stdoutdata
    assert "from an unrelated thread" not in script.stdoutdata


def test_that_concurrent_scripts_only_capture_their_own_output():
    both_are_running = threading.Barrier(2, timeout=10)

    class PrintingScript(ErtScript):
        def run(self, message):
            both_are_running.wait()
            print(message)
            both_are_running.wait()

    first, second = PrintingScript(), PrintingScript()
    threads = [
        threading.Thread(target=script.initializeAndRun, args=([str], [message]))
        for script, message in ((first, "first"), (second, "second"))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        _join(thread)

    assert first.stdoutdata.strip() == "first"
    assert second.stdoutdata.strip() == "second"


def test_that_the_original_streams_are_restored_after_capturing():
    class PrintingScript(ErtScript):
        def run(self):
            print("hello")

    stdout, stderr = sys.stdout, sys.stderr
    PrintingScript().initializeAndRun([], [])

    assert sys.stdout is stdout
    assert sys.stderr is stderr


@pytest.mark.parametrize(
    "attribute", ["fileno", "isatty", "encoding", "errors", "buffer", "line_buffering"]
)
def test_that_the_captured_stdout_exposes_the_same_attributes_as_the_real_one(
    attribute,
):
    def look_up(stream):
        """The attribute value, or the error raised, so both can be compared."""
        try:
            value = getattr(stream, attribute)
            return value() if callable(value) else value
        except Exception as e:
            return type(e)

    seen = {}

    class InspectingScript(ErtScript):
        def run(self):
            seen["value"] = look_up(sys.stdout)

    expected = look_up(sys.stdout)
    InspectingScript().initializeAndRun([], [])

    assert seen["value"] == expected
