import os
import sys
import time

import pytest

from ert._c_wrappers.job_queue import JobStatusType
from ert.simulator import BatchContext, BatchSimulator


class MockMonitor:
    def __init__(self):
        self.sim_context = None

    def start_callback(self, *args, **kwargs):
        self.sim_context = args[0]


def _wait_for_completion(ctx):
    while ctx.running():
        status = ctx.status
        time.sleep(1)
        sys.stderr.write(f"status: {status}\n")
        for job_index in range(len(ctx)):
            status = ctx.job_status(job_index)
            progress = ctx.job_progress(job_index)
            if progress:
                for job in progress.jobs:
                    sys.stderr.write(f"   {job}: \n")


@pytest.fixture
def batch_sim_example(setup_case):
    return setup_case("local/batch_sim", "batch_sim.ert")


def test_invalid_simulator_creation(batch_sim_example):
    res_config = batch_sim_example
    # Not valid ResConfig instance as first argument
    with pytest.raises(ValueError):
        rsim = BatchSimulator(
            "ARG",
            {
                "WELL_ORDER": ["W1", "W2", "W3"],
                "WELL_ON_OFF": ["W1", "W2", "W3"],
            },
            ["ORDER", "ON_OFF"],
        )

    # Control argument not a dict - Exception
    with pytest.raises(Exception):
        rsim = BatchSimulator(res_config, ["WELL_ORDER", ["W1", "W2", "W3"]], ["ORDER"])

    # Duplicate keys
    with pytest.raises(ValueError):
        rsim = BatchSimulator(res_config, {"WELL_ORDER": ["W3", "W2", "W3"]}, ["ORDER"])

    rsim = BatchSimulator(
        res_config,
        {"WELL_ORDER": ["W1", "W2", "W3"], "WELL_ON_OFF": ["W1", "W2", "W3"]},
        ["ORDER", "ON_OFF"],
    )

    # The key for one of the controls is invalid => KeyError
    with pytest.raises(KeyError):
        rsim.start(
            "case",
            [
                (
                    2,
                    {
                        "WELL_ORDERX": {"W1": 0, "W2": 0, "W3": 1},
                        "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                    },
                ),
                (
                    2,
                    {
                        "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 0},
                        "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                    },
                ),
            ],
        )

    # The key for one of the variables is invalid => KeyError
    with pytest.raises(KeyError):
        rsim.start(
            "case",
            [
                (
                    2,
                    {
                        "WELL_ORDER": {"W1": 0, "W4": 0, "W3": 1},
                        "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                    },
                ),
                (
                    1,
                    {
                        "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 0},
                        "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                    },
                ),
            ],
        )

    # The key for one of the variables is invalid => KeyError
    with pytest.raises(KeyError):
        rsim.start(
            "case",
            [
                (
                    2,
                    {
                        "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 1, "W0": 0},
                        "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                    },
                ),
                (
                    1,
                    {
                        "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 0},
                        "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                    },
                ),
            ],
        )

    # Missing the key WELL_ON_OFF => KeyError
    with pytest.raises(KeyError):
        rsim.start("case", [(2, {"WELL_ORDER": {"W1": 0, "W2": 0, "W3": 1}})])

    # One of the numeric vectors has wrong length => ValueError:
    with pytest.raises(KeyError):
        rsim.start(
            "case",
            [
                (
                    2,
                    {
                        "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 1},
                        "WELL_ON_OFF": {"W2": 0},
                    },
                ),
            ],
        )

    # Not numeric values => Exception
    with pytest.raises(Exception):
        rsim.start(
            "case",
            [
                (
                    2,
                    {
                        "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 1},
                        "WELL_ON_OFF": {"W1": 0, "W2": 1, "W3": "X"},
                    },
                ),
            ],
        )

    # Not numeric values => Exception
    with pytest.raises(Exception):
        rsim.start(
            "case",
            [
                (
                    "2",
                    {
                        "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 1},
                        "WELL_ON_OFF": {"W1": 0, "W2": 1, "W3": 4},
                    },
                ),
            ],
        )


def test_batch_simulation(batch_sim_example):
    res_config = batch_sim_example
    monitor = MockMonitor()
    rsim = BatchSimulator(
        res_config,
        {"WELL_ORDER": ["W1", "W2", "W3"], "WELL_ON_OFF": ["W1", "W2", "W3"]},
        ["ORDER", "ON_OFF"],
        callback=monitor.start_callback,
    )

    # Starting a simulation which should actually run through.
    case_data = [
        (
            2,
            {
                "WELL_ORDER": {"W1": 1, "W2": 2, "W3": 3},
                "WELL_ON_OFF": {"W1": 4, "W2": 5, "W3": 6},
            },
        ),
        (
            1,
            {
                "WELL_ORDER": {"W1": 7, "W2": 8, "W3": 9},
                "WELL_ON_OFF": {"W1": 10, "W2": 11, "W3": 12},
            },
        ),
    ]

    ctx = rsim.start("case", case_data)
    assert len(case_data) == len(ctx)

    # Asking for results before it is complete.
    with pytest.raises(RuntimeError):
        ctx.results()

    # Ask for status of simulation we do not have.
    with pytest.raises(KeyError):
        ctx.job_status(1973)

    with pytest.raises(KeyError):
        ctx.job_progress(1987)

    # Carry out simulations..
    _wait_for_completion(ctx)

    # Fetch and validate results
    results = ctx.results()
    assert len(results) == 2

    for result, (_, controls) in zip(results, case_data):
        assert sorted(result.keys()) == sorted(["ORDER", "ON_OFF"])

        for res_key, ctrl_key in (
            ("ORDER", "WELL_ORDER"),
            ("ON_OFF", "WELL_ON_OFF"),
        ):

            # The forward model job SQUARE_PARAMS will load the control
            # values and square them before writing results to disk in
            # the order W1, W2, W3.
            assert list(result[res_key]) == [
                controls[ctrl_key][var_name] ** 2 for var_name in ["W1", "W2", "W3"]
            ]

    assert isinstance(monitor.sim_context, BatchContext)


@pytest.mark.usefixtures("use_tmpdir")
def test_batch_simulation_invalid_suffixes(batch_sim_example):
    res_config = batch_sim_example

    # If suffixes are given, must be all non-empty string collections
    type_err_suffixes = (
        27,
        "astring",
        b"somebytes",
        True,
        False,
        [True, False],
        None,
        range(3),
    )
    for sfx in type_err_suffixes:
        with pytest.raises(TypeError):
            BatchSimulator(
                res_config,
                {
                    "WELL_ORDER": {"W1": ["a"], "W3": sfx},
                },
                ["ORDER"],
            )
    val_err_suffixes = (
        [],
        {},
        [""],
        ["a", "a"],
    )
    for sfx in val_err_suffixes:
        with pytest.raises(ValueError):
            BatchSimulator(
                res_config,
                {
                    "WELL_ORDER": {"W1": ["a"], "W3": sfx},
                },
                ["ORDER"],
            )

    rsim = BatchSimulator(
        res_config,
        {
            "WELL_ORDER": {
                "W1": ["a", "b"],
                "W3": ["c"],
            },
        },
        ["ORDER"],
    )

    # suffixes not taken into account
    with pytest.raises(KeyError):
        rsim.start("case", [(1, {"WELL_ORDER": {"W1": 3, "W3": 2}})])
    with pytest.raises(KeyError):
        rsim.start("case", [(1, {"WELL_ORDER": {"W1": {}, "W3": {}}})])

    # wrong suffixes
    with pytest.raises(KeyError):
        rsim.start(
            "case",
            [
                (
                    1,
                    {
                        "WELL_ORDER": {
                            "W1": {"a": 3, "x": 3},
                            "W3": {"c": 2},
                        }
                    },
                )
            ],
        )

    # missing one suffix
    with pytest.raises(KeyError):
        rsim.start(
            "case",
            [
                (
                    1,
                    {
                        "WELL_ORDER": {
                            "W1": {"a": 3},
                            "W3": {"c": 2},
                        }
                    },
                )
            ],
        )

    # wrong type for values
    # Exception cause atm this would raise a ctypes.ArgumentError
    # but that's an implementation detail that will hopefully change
    # not so far in the future
    with pytest.raises(Exception):
        rsim.start(
            "case",
            [
                (
                    1,
                    {
                        "WELL_ORDER": {
                            "W1": {"a": "3", "b": 3},
                            "W3": {"c": 2},
                        }
                    },
                )
            ],
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_batch_simulation_suffixes(batch_sim_example):
    res_config = batch_sim_example
    monitor = MockMonitor()
    rsim = BatchSimulator(
        res_config,
        {
            "WELL_ORDER": {
                "W1": ["a", "b"],
                "W2": ["c"],
                "W3": ["a", "b"],
            },
            "WELL_ON_OFF": ["W1", "W2", "W3"],
        },
        ["ORDER", "ON_OFF"],
        callback=monitor.start_callback,
    )
    # Starting a simulation which should actually run through.
    case_data = [
        (
            2,
            {
                "WELL_ORDER": {
                    "W1": {"a": 0.5, "b": 0.2},
                    "W2": {"c": 2},
                    "W3": {"a": -0.5, "b": -0.2},
                },
                "WELL_ON_OFF": {"W1": 4, "W2": 5, "W3": 6},
            },
        ),
        (
            1,
            {
                "WELL_ORDER": {
                    "W1": {"a": 0.8, "b": 0.9},
                    "W2": {"c": 1.6},
                    "W3": {"a": -0.8, "b": -0.9},
                },
                "WELL_ON_OFF": {"W1": 10, "W2": 11, "W3": 12},
            },
        ),
    ]

    ctx = rsim.start("case", case_data)
    assert len(case_data) == len(ctx)
    _wait_for_completion(ctx)

    # Fetch and validate results
    results = ctx.results()
    assert len(results) == 2

    for result in results:
        assert sorted(result.keys()) == sorted(["ORDER", "ON_OFF"])

    keys = ("W1", "W2", "W3")
    for result, (_, controls) in zip(results, case_data):
        expected = [controls["WELL_ON_OFF"][key] ** 2 for key in keys]
        assert list(result["ON_OFF"]) == expected

        expected = [
            v**2 for key in keys for _, v in controls["WELL_ORDER"][key].items()
        ]
        for exp, act in zip(expected, list(result["ORDER"])):
            assert act == pytest.approx(exp)


def test_stop_sim(batch_sim_example):
    res_config = batch_sim_example

    rsim = BatchSimulator(
        res_config,
        {"WELL_ORDER": ["W1", "W2", "W3"], "WELL_ON_OFF": ["W1", "W2", "W3"]},
        ["ORDER", "ON_OFF"],
    )

    case_name = "MyCaseName_123"

    # Starting a simulation which should actually run through.
    ctx = rsim.start(
        case_name,
        [
            (
                2,
                {
                    "WELL_ORDER": {"W1": 1, "W2": 2, "W3": 3},
                    "WELL_ON_OFF": {"W1": 4, "W2": 5, "W3": 6},
                },
            ),
            (
                1,
                {
                    "WELL_ORDER": {"W1": 7, "W2": 8, "W3": 9},
                    "WELL_ON_OFF": {"W1": 10, "W2": 11, "W3": 12},
                },
            ),
        ],
    )

    ctx.stop()
    status = ctx.status

    assert status.complete == 0
    assert status.running == 0

    runpath = f"storage/batch_sim/runpath/{case_name}/realization-0"
    assert os.path.exists(runpath)


def test_workflow_pre_simulation(batch_sim_example):
    res_config = batch_sim_example

    rsim = BatchSimulator(
        res_config,
        {"WELL_ORDER": ["W1", "W2", "W3"], "WELL_ON_OFF": ["W1", "W2", "W3"]},
        ["ORDER", "ON_OFF"],
    )

    case_name = "TestCase42"
    case_data = [
        (
            2,
            {
                "WELL_ORDER": {"W1": 1, "W2": 2, "W3": 3},
                "WELL_ON_OFF": {"W1": 4, "W2": 5, "W3": 6},
            },
        ),
        (
            1,
            {
                "WELL_ORDER": {"W1": 7, "W2": 8, "W3": 9},
                "WELL_ON_OFF": {"W1": 10, "W2": 11, "W3": 12},
            },
        ),
    ]

    # Starting a simulation which should actually run through.
    ctx = rsim.start(case_name, case_data)
    ctx.stop()

    status = ctx.status

    assert status.complete == 0
    assert status.running == 0
    for idx, _ in enumerate(case_data):
        path = (
            f"storage/batch_sim/runpath/{case_name}"
            f"/realization-{idx}/iter-0/realization.number"
        )
        assert os.path.isfile(path)
        with open(path, "r") as f:
            assert f.readline(1) == str(idx)


def assertContextStatusOddFailures(batch_ctx, final_state_only=False):
    running_status = set(
        (
            JobStatusType.JOB_QUEUE_WAITING,
            JobStatusType.JOB_QUEUE_SUBMITTED,
            JobStatusType.JOB_QUEUE_PENDING,
            JobStatusType.JOB_QUEUE_RUNNING,
            JobStatusType.JOB_QUEUE_UNKNOWN,
            JobStatusType.JOB_QUEUE_EXIT,
            JobStatusType.JOB_QUEUE_DONE,
            None,  # job is not submitted yet but ok for this test
        )
    )

    for idx in range(len(batch_ctx)):
        status = batch_ctx.job_status(idx)
        if not final_state_only and status in running_status:
            continue
        if idx % 2 == 0:
            assert status == JobStatusType.JOB_QUEUE_SUCCESS
        else:
            assert status == JobStatusType.JOB_QUEUE_FAILED


@pytest.mark.usefixtures("use_tmpdir")
def test_batch_ctx_status_failing_jobs(setup_case):

    res_config = setup_case("local/batch_sim", "batch_sim_sleep_and_fail.ert")

    external_parameters = {
        "WELL_ORDER": ("W1", "W2", "W3"),
        "WELL_ON_OFF": ("W1", "W2", "W3"),
    }
    results = ("ORDER", "ON_OFF")
    rsim = BatchSimulator(res_config, external_parameters, results)

    cases = [
        (
            0,
            {
                "WELL_ORDER": {"W1": idx + 1, "W2": idx + 2, "W3": idx + 3},
                "WELL_ON_OFF": {"W1": idx * 4, "W2": idx * 5, "W3": idx * 6},
            },
        )
        for idx in range(10)
    ]

    batch_ctx = rsim.start("case_name", cases)
    while batch_ctx.running():
        assertContextStatusOddFailures(batch_ctx)
        time.sleep(1)

    assertContextStatusOddFailures(batch_ctx, final_state_only=True)
