import threading
import time

import numpy as np
import pytest
import resfo

from tests.ert.utils import SOURCE_DIR

from ._import_from_location import import_from_location

# import ecl_config.py and ecl_run from ert/resources/forward_models
# package-data path which. These are kept out of the ert package to avoid the
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


def test_runners_are_found_from_path():
    # assert different runners are looked for given wanted
    # simulator
    # assert runtimeoerror if no runner found
    pass


def test_flowrun_can_be_bypassed():
    # if flow is in path, then we can bypass
    # assert an error if num_cpu is more than 1., not suppported yet.
    pass


def test_runner_fails_on_missing_data_file():
    pass


def test_ecl_case_from_data_file():
    pass


@pytest.mark.usefixtures("use_tmpdir")
def test_await_completed_summary_file_will_timeout_on_missing_smry():
    assert (
        # Expected wait time is 0.3
        run_reservoirsimulator.await_completed_unsmry_file(
            "SPE1.UNSMRY", max_wait=0.3, poll_interval=0.1
        )
        > 0.3
    )


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
