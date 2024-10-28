import os
import sys
import time

import pytest

from ert.config import ErtConfig, ExtParamConfig, GenDataConfig
from ert.scheduler import JobState
from ert.simulator import BatchContext, BatchSimulator


class MockMonitor:
    def __init__(self):
        self.sim_context = None

    def start_callback(self, *args, **kwargs):
        self.sim_context = args[0]


def _wait_for_completion(ctx: BatchContext):
    while ctx.running():
        status = ctx.status
        time.sleep(1)
        sys.stderr.write(f"status: {status}\n")
        for job_index in range(len(ctx)):
            status = ctx.get_job_state(job_index)
            progress = ctx.job_progress(job_index)
            if progress:
                for job in progress.steps:
                    sys.stderr.write(f"   {job}: \n")


@pytest.fixture
def batch_sim_example(setup_case):
    return setup_case("batch_sim", "batch_sim.ert")


# The batch simulator was recently refactored. It now requires an ERT config
# object that has been generated in the derived Simulator class. The resulting
# ERT config object includes features that cannot be specified in an ERT
# configuration file. This is acceptable since the batch simulator is only used
# by Everest and slated to be replaced in the near future with newer ERT
# functionality. However, the tests in this file assume that the batch simulator
# can be configured independently from an Everest configuration. To make the
# tests work, the batch simulator class is patched here to inject the missing
# functionality.
class PatchedBatchSimulator(BatchSimulator):
    def __init__(self, ert_config, storage, controls, results, callback=None):
        try:
            ens_config = ert_config.ensemble_config
            for control_name, variables in controls.items():
                ens_config.addNode(
                    ExtParamConfig(
                        name=control_name,
                        input_keys=variables,
                        output_file=control_name + ".json",
                    )
                )

            if "gen_data" not in ens_config:
                ens_config.addNode(
                    GenDataConfig(
                        keys=results,
                        input_files=[f"{k}" for k in results],
                        report_steps_list=[None for _ in results],
                    )
                )
            else:
                existing_gendata = ens_config.response_configs["gen_data"]
                existing_keys = existing_gendata.keys
                assert isinstance(existing_gendata, GenDataConfig)

                for key in results:
                    if key not in existing_keys:
                        existing_gendata.keys.append(key)
                        existing_gendata.input_files.append(f"{key}")
                        existing_gendata.report_steps_list.append(None)

            experiment = storage.create_experiment(
                name="EnOptCase",
                parameters=ens_config.parameter_configuration,
                responses=ens_config.response_configuration,
            )
            super().__init__(
                perferred_num_cpu=ert_config.preferred_num_cpu,
                runpath_file=ert_config.runpath_file,
                user_config_file=ert_config.user_config_file,
                env_vars=ert_config.env_vars,
                forward_model_steps=ert_config.forward_model_steps,
                parameter_configurations=ert_config.ensemble_config.parameter_configs,
                substitutions=ert_config.substitutions,
                queue_config=ert_config.queue_config,
                model_config=ert_config.model_config,
                analysis_config=ert_config.analysis_config,
                hooked_workflows=ert_config.hooked_workflows,
                templates=ert_config.ert_templates,
                experiment=experiment,
                controls=set(controls),
                results=results,
                callback=callback,
            )
        except Exception as e:
            raise e


def test_that_batch_simulator_gives_good_message_on_duplicate_keys(
    minimum_case, storage
):
    with pytest.raises(ValueError, match="Duplicate keys"):
        _ = PatchedBatchSimulator(
            minimum_case, storage, {"WELL_ORDER": ["W3", "W2", "W3"]}, ["ORDER"]
        )


@pytest.fixture
def batch_simulator(batch_sim_example, storage):
    return PatchedBatchSimulator(
        batch_sim_example,
        storage,
        {"WELL_ORDER": ["W1", "W2", "W3"], "WELL_ON_OFF": ["W1", "W2", "W3"]},
        ["ORDER", "ON_OFF"],
    )


@pytest.mark.parametrize(
    "_input, match",
    [
        (
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
            "Mismatch between initialized and provided",
        ),
        (
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
            "No such key: W4",
        ),
        (
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
            "Expected 3 variables",
        ),
        (
            [(2, {"WELL_ORDER": {"W1": 0, "W2": 0, "W3": 1}})],
            "Mismatch between initialized and provided",
        ),
        (
            [
                (
                    2,
                    {
                        "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 1},
                        "WELL_ON_OFF": {"W2": 0},
                    },
                ),
            ],
            "Expected 3 variables",
        ),
    ],
)
def test_that_starting_with_invalid_key_raises_key_error(
    batch_simulator, _input, match, storage
):
    with pytest.raises(KeyError, match=match):
        batch_simulator.start("case", _input)


@pytest.mark.integration_test
def test_batch_simulation(batch_simulator, storage):
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

    ctx = batch_simulator.start("case", case_data)
    assert len(case_data) == len(ctx.mask)

    # Asking for results before it is complete.
    with pytest.raises(RuntimeError):
        ctx.results()
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
            # The forward model step SQUARE_PARAMS will load the control
            # values and square them before writing results to disk in
            # the order W1, W2, W3.
            assert list(result[res_key]) == [
                controls[ctrl_key][var_name] ** 2 for var_name in ["W1", "W2", "W3"]
            ]


@pytest.mark.parametrize(
    "suffix, error",
    (
        (27, TypeError),
        ("astring", TypeError),
        (b"somebytes", TypeError),
        (True, TypeError),
        (False, TypeError),
        ([True, False], TypeError),
        (None, TypeError),
        (range(3), TypeError),
        ([], ValueError),
        ({}, TypeError),
        ([""], ValueError),
        (["a", "a"], ValueError),
    ),
)
def test_that_batch_simulation_handles_invalid_suffixes_at_init(
    batch_sim_example, suffix, error, storage
):
    with pytest.raises(error):
        _ = PatchedBatchSimulator(
            batch_sim_example,
            storage,
            {
                "WELL_ORDER": {"W1": ["a"], "W3": suffix},
            },
            ["ORDER"],
        )


@pytest.mark.parametrize(
    "inp, match",
    [
        ([(1, {"WELL_ORDER": {"W1": 3, "W3": 2}})], "Key W1 has suffixes"),
        ([(1, {"WELL_ORDER": {"W1": {}, "W3": {}}})], "Key W1 is missing"),
        (
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
            "Key W1 has suffixes",
        ),
        (
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
            "Key W1 is missing",
        ),
    ],
)
def test_that_batch_simulator_handles_invalid_suffixes_at_start(
    batch_sim_example, inp, match, storage
):
    rsim = PatchedBatchSimulator(
        batch_sim_example,
        storage,
        {
            "WELL_ORDER": {
                "W1": ["a", "b"],
                "W3": ["c"],
            },
        },
        ["ORDER"],
    )

    with pytest.raises(KeyError, match=match):
        rsim.start("case", inp)


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_batch_simulation_suffixes(batch_sim_example, storage):
    ert_config = batch_sim_example
    monitor = MockMonitor()
    rsim = PatchedBatchSimulator(
        ert_config,
        storage,
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

        # [:3] slicing can be removed when responses are not stored in netcdf leading
        # to redundant nans from combining->selecting
        assert list(result["ON_OFF"][:3]) == expected

        expected = [
            v**2 for key in keys for _, v in controls["WELL_ORDER"][key].items()
        ]
        for exp, act in zip(expected, list(result["ORDER"])):
            assert act == pytest.approx(exp)


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=3)  # https://github.com/equinor/ert/issues/7309
@pytest.mark.timeout(10)
def test_stop_sim(copy_case, storage):
    copy_case("batch_sim")
    with open("sleepy_time.ert", "a", encoding="utf-8") as f:
        f.write(
            """
LOAD_WORKFLOW_JOB workflows/jobs/REALIZATION_NUMBER
LOAD_WORKFLOW workflows/REALIZATION_NUMBER_WORKFLOW
HOOK_WORKFLOW REALIZATION_NUMBER_WORKFLOW PRE_SIMULATION
LOAD_WORKFLOW_JOB workflows/jobs/REALIZATION_NUMBER
        """
        )

    ert_config = ErtConfig.from_file("sleepy_time.ert")

    rsim = PatchedBatchSimulator(
        ert_config,
        storage,
        {"WELL_ORDER": ["W1", "W2", "W3"], "WELL_ON_OFF": ["W1", "W2", "W3"]},
        ["ORDER", "ON_OFF"],
    )

    case_name = "MyCaseName_123"
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

    paths = (
        "runpath/realization-0-2/iter-0/realization.number",
        "runpath/realization-1-1/iter-0/realization.number",
    )
    for idx, path in enumerate(paths):
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as f:
            assert f.readline(1) == str(idx)


def assertContextStatusOddFailures(batch_ctx: BatchContext, final_state_only=False):
    running_status = {
        JobState.WAITING,
        JobState.SUBMITTING,
        JobState.PENDING,
        JobState.RUNNING,
        JobState.ABORTING,
        JobState.ABORTED,
        JobState.FAILED,
        JobState.COMPLETED,
        None,  # job is not submitted yet but ok for this test
    }

    for idx in range(len(batch_ctx)):
        status = batch_ctx.get_job_state(idx)
        if not final_state_only and status in running_status:
            continue
        if idx % 2 == 0:
            assert status == JobState.COMPLETED
        else:
            assert status == JobState.FAILED


@pytest.mark.integration_test
def test_batch_ctx_status_failing_jobs(setup_case, storage):
    ert_config = setup_case("batch_sim", "batch_sim_sleep_and_fail.ert")

    external_parameters = {
        "WELL_ORDER": ("W1", "W2", "W3"),
        "WELL_ON_OFF": ("W1", "W2", "W3"),
    }
    results = ("ORDER", "ON_OFF")
    rsim = PatchedBatchSimulator(ert_config, storage, external_parameters, results)

    ensembles = [
        (
            0,
            {
                "WELL_ORDER": {"W1": idx + 1, "W2": idx + 2, "W3": idx + 3},
                "WELL_ON_OFF": {"W1": idx * 4, "W2": idx * 5, "W3": idx * 6},
            },
        )
        for idx in range(10)
    ]

    batch_ctx = rsim.start("case_name", ensembles)
    while batch_ctx.running():
        assertContextStatusOddFailures(batch_ctx)
        time.sleep(1)

    assertContextStatusOddFailures(batch_ctx, final_state_only=True)
