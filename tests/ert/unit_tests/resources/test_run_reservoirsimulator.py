import os
import shutil
import stat
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import resfo

from tests.ert.utils import SOURCE_DIR

from ._import_from_location import import_from_location

# import run_reservoirsimulator from ert/resources/forward_models
# package-data. This is kept out of the ert package to avoid the
# overhead of importing ert. This is necessary as these may be invoked as a
# subprocess on each realization.


run_reservoirsimulator = import_from_location(
    "run_reservoirsimulator",
    SOURCE_DIR / "src/ert/resources/forward_models/run_reservoirsimulator.py",
)
ecl_case_to_data_file = import_from_location(
    "ecl_case_to_data_file",
    SOURCE_DIR / "src/ert/resources/forward_models/run_reservoirsimulator.py",
)


@pytest.fixture(name="mocked_eclrun")
def fixture_mocked_eclrun(use_tmpdir, monkeypatch):
    """This puts a eclrun binary in path that cannot do anything."""
    eclrun_bin = Path("bin/eclrun")
    eclrun_bin.parent.mkdir()
    eclrun_bin.write_text("", encoding="utf-8")
    eclrun_bin.chmod(eclrun_bin.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"bin:{os.environ['PATH']}")


def test_unknown_simulator():
    with pytest.raises(ValueError, match="Unknown simulator"):
        run_reservoirsimulator.RunReservoirSimulator(
            "bogus_flow", "mocked_version", "bogus_deck.DATA"
        )


@pytest.mark.usefixtures("mocked_eclrun")
def test_runner_fails_on_missing_data_file():
    with pytest.raises(OSError, match=r"No such file: NOTEXISTING\.DATA"):
        run_reservoirsimulator.RunReservoirSimulator(
            "eclipse", "mocked_version", "NOTEXISTING.DATA"
        )


@pytest.mark.usefixtures("mocked_eclrun")
def test_runner_can_find_deck_without_extension():
    Path("DECK.DATA").write_text("FOO", encoding="utf-8")
    runner = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "mocked_version", "DECK"
    )
    assert runner.data_file == "DECK.DATA"


@pytest.mark.usefixtures("mocked_eclrun")
def test_runner_can_find_lowercase_deck_without_extension():
    Path("deck.data").write_text("FOO", encoding="utf-8")
    runner = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "mocked_version", "deck"
    )
    assert runner.data_file == "deck.data"


@pytest.mark.skipif(
    sys.platform.startswith("darwin"), reason="Case insensitive filesystem on MacOS"
)
@pytest.mark.usefixtures("mocked_eclrun")
def test_runner_cannot_find_mixed_case_decks():
    Path("deck.DATA").write_text("FOO", encoding="utf-8")
    with pytest.raises(OSError, match=r"No such file: deck\.data"):
        run_reservoirsimulator.RunReservoirSimulator(
            "eclipse", "mocked_version", "deck"
        )


@pytest.mark.usefixtures("mocked_eclrun")
@pytest.mark.parametrize(
    "data_path, expected",
    [
        ("DECK.DATA", "DECK"),
        ("foo/DECK.DATA", "DECK"),
        ("foo/deck.data", "deck"),
    ],
)
def test_runner_can_extract_base_name(data_path: str, expected: str):
    Path(data_path).parent.mkdir(exist_ok=True)
    Path(data_path).write_text("FOO", encoding="utf-8")
    runner = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "mocked_version", data_path
    )
    assert runner.base_name == expected


@pytest.mark.usefixtures("use_tmpdir")
def test_await_completed_summary_file_will_return_asap():
    resfo.write("FOO.UNSMRY", [("INTEHEAD", np.array([1], dtype=np.int32))])
    assert (
        0.01
        # Expected wait time is the poll_interval
        < run_reservoirsimulator.await_completed_unsmry_file(
            "FOO.UNSMRY", max_wait=0.5, poll_interval=0.1
        )
        < 0.4
    )


@pytest.mark.flaky(reruns=5)
@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_await_completed_summary_file_will_wait_for_slow_smry():
    # This is a timing test, and has inherent flakiness:
    #  * Reading and writing to the same smry file at the same time
    #    can make the reading throw an exception every time, and can
    #    result in max_wait triggering.
    #  * If the writer thread is starved, two consecutive polls may
    #    yield the same summary length, resulting in premature exit.
    #  * Heavily loaded hardware can make everything go too slow.
    def slow_smry_writer():
        for size in range(10):
            resfo.write(
                "FOO.UNSMRY", (size + 1) * [("INTEHEAD", np.array([1], dtype=np.int32))]
            )
            time.sleep(0.05)

    thread = threading.Thread(target=slow_smry_writer)
    thread.start()
    time.sleep(0.1)  # Let the thread start writing
    assert (
        0.5
        # Minimal wait time is around 0.55
        < run_reservoirsimulator.await_completed_unsmry_file(
            "FOO.UNSMRY", max_wait=4, poll_interval=0.21
        )
        < 2
    )
    thread.join()


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize("error_count", [0, 1])
def test_encoding_errors_in_prt_are_ignored(monkeypatch, error_count):
    eclrun_bin = Path("bin/eclrun")
    eclrun_bin.parent.mkdir()
    eclrun_bin.write_text("#!/bin/sh\necho $@ > eclrun_args\nexit 0", encoding="utf-8")
    eclrun_bin.chmod(eclrun_bin.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"bin:{os.environ['PATH']}")

    Path("EIGHTCELLS.DATA").touch()
    Path("EIGHTCELLS.PRT").write_text(
        f"æøå\nErrors {error_count}\nBugs 0", encoding="latin1"
    )

    with pytest.raises(SystemExit) as excinfo:  # noqa: PT012
        run_reservoirsimulator.run_reservoirsimulator(
            [
                "eclipse",
                "EIGHTCELLS.DATA",
                "--version",
                "2019.3",
            ]
        )
        raise SystemExit(0)

    # when error_count is 0 or 1, this will work:
    assert excinfo.value.code == -error_count


@pytest.mark.parametrize("simulator", ["eclipse", "e300", "flow"])
@pytest.mark.usefixtures("use_tmpdir")
def test_runner_will_forward_unknown_arguments(monkeypatch, simulator):
    eclrun_bin = Path("bin/eclrun")
    eclrun_bin.parent.mkdir()
    eclrun_bin.write_text("#!/bin/sh\necho $@ > eclrun_args\nexit 0", encoding="utf-8")
    eclrun_bin.chmod(eclrun_bin.stat().st_mode | stat.S_IEXEC)
    shutil.copy(eclrun_bin, "bin/flowrun")
    monkeypatch.setenv("PATH", f"bin:{os.environ['PATH']}")
    Path("EIGHTCELLS.DATA").touch()
    Path("EIGHTCELLS.PRT").write_text("Errors 0\nBugs 0", encoding="utf-8")

    run_reservoirsimulator.run_reservoirsimulator(
        [
            simulator,
            "EIGHTCELLS.DATA",
            "--version",
            "2019.3",
            "--arbitrary_option_being_forwarded",
        ]
    )

    assert "--arbitrary_option_being_forwarded" in Path("eclrun_args").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("simulator", ["eclipse", "e300", "flow"])
@pytest.mark.usefixtures("use_tmpdir")
def test_runner_will_forward_multiple_unknown_arguments(monkeypatch, simulator):
    eclrun_bin = Path("bin/eclrun")
    eclrun_bin.parent.mkdir()
    eclrun_bin.write_text(
        '#!/bin/sh\nfor arg in "$@"; do echo $arg >> eclrun_args; done\nexit 0',
        encoding="utf-8",
    )
    eclrun_bin.chmod(eclrun_bin.stat().st_mode | stat.S_IEXEC)
    shutil.copy(eclrun_bin, "bin/flowrun")
    monkeypatch.setenv("PATH", f"bin:{os.environ['PATH']}")
    Path("EIGHTCELLS.DATA").touch()
    Path("EIGHTCELLS.PRT").write_text("Errors 0\nBugs 0", encoding="utf-8")

    run_reservoirsimulator.run_reservoirsimulator(
        [
            simulator,
            "EIGHTCELLS.DATA",
            "--version",
            "2019.3",
            "--first-option --second-option",
        ]
    )

    assert "--first-option\n--second-option" in Path("eclrun_args").read_text(
        encoding="utf-8"
    )
