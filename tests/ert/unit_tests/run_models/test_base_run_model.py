import asyncio
import math
import os
import re
import uuid
from logging import Logger
from pathlib import Path
from queue import SimpleQueue
from types import MethodType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ConfigDict

from ert.config import ErtConfig, GenKwConfig, ModelConfig, QueueConfig, QueueSystem
from ert.config.queue_config import LsfQueueOptions
from ert.ensemble_evaluator import EndEvent, EvaluatorServerConfig
from ert.ensemble_evaluator.snapshot import EnsembleSnapshot
from ert.plugins import ErtRuntimePlugins
from ert.run_models.run_model import RunModel, UserCancelled


@pytest.fixture(autouse=True)
def patch_abstractmethods(monkeypatch):
    monkeypatch.setattr(RunModel, "__abstractmethods__", set())


class MockJob:
    def __init__(self, status) -> None:
        self.status = status


def create_run_model(**kwargs):
    default_args = {
        # Note: Will create a storage in cwd
        "storage_path": "./storage",
        "runpath_file": MagicMock(spec=Path),
        "user_config_file": MagicMock(spec=Path),
        "env_vars": MagicMock(spec=dict),
        "env_pr_fm_step": MagicMock(spec=dict),
        "runpath_config": ModelConfig(),
        "queue_config": MagicMock(spec=QueueConfig),
        "forward_model_steps": MagicMock(spec=list),
        "status_queue": MagicMock(spec=SimpleQueue),
        "substitutions": MagicMock(spec=dict),
        "hooked_workflows": MagicMock(spec=dict),
        "active_realizations": MagicMock(spec=list),
        "random_seed": 123,
        "log_path": Path(""),
    }

    class RunModelWithMockSupport(RunModel):
        model_config = ConfigDict(frozen=False, extra="allow")

    return RunModelWithMockSupport(**(default_args | kwargs))


def test_run_model_does_not_support_rerun_failed_realizations(minimum_case):
    brm = create_run_model(
        storage_path=minimum_case.ens_path,
        queue_config=minimum_case.queue_config,
        active_realizations=[True],
        forward_model_steps=minimum_case.forward_model_steps,
    )
    assert not brm.supports_rerunning_failed_realizations


def test_status_when_rerunning_on_non_rerunnable_model(use_tmpdir):
    brm = create_run_model()
    brm._status_queue = SimpleQueue()
    brm.start_simulations_thread(
        EvaluatorServerConfig(use_token=False), rerun_failed_realizations=True
    )
    assert brm._status_queue.get() == EndEvent(
        event_type="EndEvent",
        failed=True,
        msg="Run model None does not support restart/rerun of failed simulations.\n",
    )


@pytest.mark.parametrize(
    "initials",
    [
        ([]),
        ([True]),
        ([False]),
        ([False, True]),
        ([True, True]),
        ([False, True]),
    ],
)
def test_active_realizations(initials, use_tmpdir):
    brm = create_run_model(active_realizations=initials)
    brm._initial_realizations_mask = initials
    assert brm.ensemble_size == len(initials)


@pytest.mark.parametrize(
    "initials, completed, any_failed, failures",
    [
        ([True], [False], True, [True]),
        ([False], [False], False, [False]),
        ([False, True], [True, False], True, [False, True]),
        ([False, True], [False, True], False, [False, False]),
        ([False, False], [False, False], False, [False, False]),
        ([False, False], [True, True], False, [False, False]),
        ([True, True], [False, True], True, [True, False]),
    ],
)
def test_failed_realizations(initials, completed, any_failed, failures, use_tmpdir):
    brm = create_run_model(active_realizations=initials)
    brm._initial_realizations_mask = initials
    brm._completed_realizations_mask = completed

    assert brm._create_mask_from_failed_realizations() == failures
    assert brm.has_failed_realizations() == any_failed


@pytest.mark.parametrize(
    "run_path, number_of_iterations, start_iteration, "
    "active_realizations_mask, expected",
    [
        ("out/realization-%d/iter-%d", 4, 2, [True, True, True, True], False),
        ("out/realization-%d/iter-%d", 4, 1, [True, True, True, True], True),
        ("out/realization-%d/iter-%d", 4, 1, [False, False, True, False], False),
        ("out/realization-%d/iter-%d", 4, 0, [False, False, False, False], False),
        ("out/realization-%d/iter-%d", 4, 0, [], False),
        ("out/realization-%d", 2, 1, [False, True, True], True),
        ("out/realization-%d", 2, 0, [False, False, True], False),
    ],
)
def test_check_if_runpath_exists_with_substitutions(
    create_dummy_run_path,
    run_path: str,
    number_of_iterations: int,
    start_iteration: int,
    active_realizations_mask: list,
    expected: bool,
    use_tmpdir,
):
    model_config = ModelConfig(runpath_format_string=run_path)
    brm = create_run_model(
        runpath_config=model_config,
        substitutions={},
        active_realizations=active_realizations_mask,
        start_iteration=start_iteration,
        _total_iterations=number_of_iterations,
    )
    assert brm.check_if_runpath_exists() == expected


@pytest.mark.parametrize(
    "active_realizations_mask, expected_number",
    [
        ([True, True, True, True], 2),
        ([False, False, True, True], 0),
        ([True, False, False, True], 1),
    ],
)
def test_get_number_of_existing_runpaths(
    create_dummy_run_path,
    active_realizations_mask,
    expected_number,
):
    run_path = "out/realization-%d/iter-%d"
    model_config = ModelConfig(runpath_format_string=run_path)
    brm = create_run_model(
        runpath_config=model_config,
        substitutions={},
        active_realizations=active_realizations_mask,
    )

    assert brm.get_number_of_existing_runpaths() == expected_number


@pytest.mark.parametrize(
    "run_path_format",
    ["<ERTCASE>/realization-<IENS>/iter-<ITER>", "<ERTCASE>/realization-<IENS>"],
)
@pytest.mark.parametrize(
    "active_realizations", [[True], [True, True], [True, False], [False], [False, True]]
)
def test_delete_run_path(run_path_format, active_realizations, use_tmpdir):
    expected_remaining = []
    expected_removed = []
    for iens, mask in enumerate(active_realizations):
        run_path = Path(
            run_path_format.replace("<IENS>", str(iens))
            .replace("<ITER>", "0")
            .replace("<ERTCASE>", "Case_Name")
        )
        os.makedirs(run_path)
        assert run_path.exists()
        if not mask:
            expected_remaining.append(run_path)
        else:
            expected_removed.append(run_path)
    share_path = Path("share")
    os.makedirs(share_path)
    model_config = ModelConfig(runpath_format_string=run_path_format)

    brm = create_run_model(
        runpath_config=model_config,
        substitutions={"<ITER>": "0", "<ERTCASE>": "Case_Name"},
        active_realizations=active_realizations,
    )

    brm.rm_run_path()
    assert not any(path.exists() for path in expected_removed)
    assert all(path.parent.exists() for path in expected_removed)
    assert all(path.exists() for path in expected_remaining)
    assert share_path.exists()


def test_num_cpu_is_propagated_from_config_to_ensemble(run_args, use_tmpdir):
    # Given NUM_CPU in the config file has a special value
    config = ErtConfig.from_file_contents("NUM_REALIZATIONS 2\nNUM_CPU 42")
    # Set up a RunModel object from the config above:

    brm = create_run_model(
        queue_config=config.queue_config,
        substitutions=config.substitutions,
        active_realizations=[True],
    )

    run_args = run_args(config, MagicMock())

    # Instead of running the RunModel, we only test its implementation detail
    # which is to use _build_ensemble() just prior to running
    ensemble = brm._build_ensemble(run_args, uuid.uuid1())

    # Assert the built ensemble has the correct NUM_CPU information
    assert ensemble.reals[0].num_cpu == 42
    assert ensemble.reals[1].num_cpu == 42


@pytest.mark.parametrize(
    "real_status_dict, expected_result",
    [
        pytest.param(
            {"0": "Finished", "1": "Finished", "2": "Finished"},
            {"Finished": 3},
            id="ran_all_realizations_and_all_succeeded",
        ),
        pytest.param(
            {"0": "Finished", "1": "Finished", "2": "Failed"},
            {"Finished": 2, "Failed": 1},
            id="ran_all_realizations_and_some_failed",
        ),
        pytest.param(
            {"0": "Finished", "1": "Running", "2": "Failed"},
            {"Finished": 1, "Failed": 1, "Running": 1},
            id="ran_all_realizations_and_result_was_mixed",
        ),
    ],
)
def test_get_current_status(
    real_status_dict,
    expected_result,
    use_tmpdir,
):
    config = ErtConfig.from_file_contents("NUM_REALIZATIONS 3")
    initial_active_realizations = [True] * 3
    new_active_realizations = [True] * 3

    brm = create_run_model(
        queue_config=config.queue_config,
        substitutions=config.substitutions,
        active_realizations=initial_active_realizations,
    )

    snapshot_dict_reals = {}
    for index, realization_status in real_status_dict.items():
        snapshot_dict_reals[index] = {"status": realization_status}
    iter_snapshot = EnsembleSnapshot.from_nested_dict({"reals": snapshot_dict_reals})
    brm._iter_snapshot[0] = iter_snapshot
    brm.active_realizations = new_active_realizations
    assert dict(brm.get_current_status()) == expected_result


@pytest.mark.parametrize(
    "initial_active_realizations, new_active_realizations, "
    "real_status_dict, expected_result",
    [
        pytest.param(
            [True, True, True],
            [False, False, False],
            {},
            {"Finished": 3},
            id="all_realizations_in_previous_run_succeeded",
        ),
        pytest.param(
            [True, True, True],
            [False, True, False],
            {},
            {"Finished": 2},
            id="some_realizations_in_previous_run_succeeded",
        ),
        pytest.param(
            [True, True, True],
            [True, True, True],
            {},
            {"Finished": 0},
            id="no_realizations_in_previous_run_succeeded",
        ),
        pytest.param(
            [False, True, True],
            [False, False, True],
            {},
            {"Finished": 1},
            id="did_not_run_all_realizations_and_some_succeeded",
        ),
        pytest.param(
            [False, True, True],
            [False, True, True],
            {},
            {"Finished": 0},
            id="did_not_run_all_realizations_and_none_succeeded",
        ),
        pytest.param(
            [True, True, True],
            [True, True, False],
            {"0": "Finished", "1": "Finished"},
            {"Finished": 3},
            id="reran_some_realizations_and_all_finished",
        ),
        pytest.param(
            [False, True, True],
            [False, True, False],
            {"1": "Finished"},
            {"Finished": 2},
            id="did_not_run_all_realizations_then_reran_and_the_realizations_finished",
        ),
    ],
)
def test_get_current_status_when_rerun(
    initial_active_realizations,
    new_active_realizations,
    real_status_dict: dict[str, str],
    expected_result,
    use_tmpdir,
):
    """Active realizations gets changed when we choose to rerun, and the result from
    the previous run should be included in the current_status."""
    config = ErtConfig.from_file_contents("NUM_REALIZATIONS 3")
    brm = create_run_model(
        queue_config=config.queue_config,
        substitutions=config.substitutions,
        active_realizations=initial_active_realizations,
    )

    brm._is_rerunning_failed_realizations = True
    snapshot_dict_reals = {}
    for index, realization_status in real_status_dict.items():
        snapshot_dict_reals[index] = {"status": realization_status}
    iter_snapshot = EnsembleSnapshot.from_nested_dict({"reals": snapshot_dict_reals})
    brm._iter_snapshot[0] = iter_snapshot
    brm.active_realizations = new_active_realizations
    assert dict(brm.get_current_status()) == expected_result


def test_get_current_status_for_new_iteration_when_realization_failed_in_previous_run(
    use_tmpdir,
):
    """Active realizations gets changed when we run next iteration, and the failed
    realizations from the previous run should not be present in the current_status."""
    initial_active_realizations = [True] * 5
    # Realization 0,1, and 3 failed in the previous iteration
    new_active_realizations = [False, False, True, False, True]
    config = ErtConfig.from_file_contents("NUM_REALIZATIONS 5")

    brm = create_run_model(
        queue_config=config.queue_config,
        substitutions=config.substitutions,
        active_realizations=initial_active_realizations,
    )

    snapshot_dict_reals = {
        "2": {"status": "Running"},
        "4": {"status": "Finished"},
    }
    iter_snapshot = EnsembleSnapshot.from_nested_dict({"reals": snapshot_dict_reals})
    brm._iter_snapshot[0] = iter_snapshot
    brm.active_realizations = new_active_realizations

    assert brm._is_rerunning_failed_realizations is False
    assert dict(brm.get_current_status()) == {"Running": 1, "Finished": 1}


@pytest.mark.parametrize(
    "new_active_realizations, was_rerun, expected_result",
    [
        pytest.param(
            [False, False, False, True, False],
            True,
            5,
            id="rerun_so_total_realization_count_is_not_affected_by_previous_failed_realizations",
        ),
        pytest.param(
            [True, True, False, False, False],
            False,
            2,
            id="new_iteration_so_total_realization_count_is_only_previously_successful_realizations",
        ),
    ],
)
def test_get_number_of_active_realizations_varies_when_rerun_or_new_iteration(
    new_active_realizations, was_rerun, expected_result, use_tmpdir
):
    """When rerunning, we include all realizations in the total amount of active
    realization. When running a new iteration based on the result of the previous
    iteration, we only include the successful realizations."""
    initial_active_realizations = [True] * 5
    config = ErtConfig.from_file_contents("NUM_REALIZATIONS 5")

    brm = create_run_model(
        queue_config=config.queue_config,
        substitutions=config.substitutions,
        active_realizations=initial_active_realizations,
    )

    brm.active_realizations = new_active_realizations
    brm._is_rerunning_failed_realizations = was_rerun
    assert brm.get_number_of_active_realizations() == expected_result


async def test_terminate_in_pre_evaluation(use_tmpdir):
    brm = create_run_model()
    brm._end_event.set()
    with pytest.raises(
        UserCancelled, match="Experiment cancelled by user in pre evaluation"
    ):
        await brm.run_ensemble_evaluator_async(AsyncMock(), AsyncMock(), AsyncMock())


@patch("ert.run_models.run_model.EnsembleEvaluator")
async def test_terminate_in_post_evaluation(evaluator, use_tmpdir):
    async def mocked_run_and_get_successful_realizations() -> list[int]:
        return list(range(5))

    evaluator().run_and_get_successful_realizations = (
        mocked_run_and_get_successful_realizations
    )
    evaluator()._server_started = asyncio.Future()
    evaluator()._server_started.set_result(None)

    async def send_terminate(end_event) -> bool:
        end_event.set()
        return True

    brm = create_run_model()
    evaluator().wait_for_evaluation_result = MethodType(send_terminate, brm._end_event)
    with pytest.raises(
        UserCancelled,
        match="Experiment cancelled by user in post evaluation",
    ):
        await brm.run_ensemble_evaluator_async(AsyncMock(), AsyncMock(), AsyncMock())


@pytest.mark.parametrize(
    "real_status_dict, current_iteration, start_iteration,"
    " total_iterations, expected_result",
    [
        pytest.param(
            {"0": "Finished", "1": "Running", "2": "Failed"},
            1,
            1,
            1,
            0.67,
            id="progress_with_single_offset_iteration",
        ),
        pytest.param(
            {"0": "Finished", "1": "Running", "2": "Running"},
            0,
            0,
            1,
            0.33,
            id="progress_with_partial_completed",
        ),
        pytest.param(
            {"0": "Running", "1": "Running", "2": "Running"},
            0,
            0,
            1,
            0.0,
            id="progress_with_none_finished",
        ),
        pytest.param(
            {"0": "Finished", "1": "Failed", "2": "Finished"},
            0,
            0,
            1,
            1.0,
            id="progress_with_all_completed",
        ),
        pytest.param(
            {"0": "Finished", "1": "Finished", "2": "Running"},
            2,
            2,
            3,
            0.22,
            id="progress_with_extended_offset_iterations",
        ),
        pytest.param(
            {"0": "Finished", "1": "Finished", "2": "Running"},
            3,
            2,
            3,
            0.55,
            id="progress_with_extended_offset_iterations",
        ),
        pytest.param(
            {"0": "Finished", "1": "Finished", "2": "Running"},
            5,
            3,
            7,
            0.38,
            id="progress_with_another_extended_offset_iterations",
        ),
    ],
)
def test_progress_calculations(
    real_status_dict: dict[str, str],
    current_iteration: int,
    start_iteration: int,
    total_iterations: int,
    expected_result: float,
    use_tmpdir,
):
    brm = create_run_model(
        start_iteration=start_iteration,
        _total_iterations=total_iterations,
        active_realizations=[True] * len(real_status_dict),
    )

    for i in range(start_iteration, start_iteration + total_iterations):
        snapshot_dict_reals = {}

        for index, realization_status in real_status_dict.items():
            status = realization_status if i == current_iteration else "Finished"
            snapshot_dict_reals[index] = {"status": status}

        iter_snapshot = EnsembleSnapshot.from_nested_dict(
            {"reals": snapshot_dict_reals}
        )
        brm._iter_snapshot[i] = iter_snapshot

        if i == current_iteration:
            break

    progress = brm.calculate_current_progress()
    assert math.isclose(progress, expected_result, abs_tol=0.1)


@pytest.mark.parametrize(
    "active_mask, expected",
    [
        ([True, True, True, True], True),
        ([False, False, True, False], False),
        ([], False),
        ([False, True, True], True),
        ([False, False, True], False),
    ],
)
def test_check_if_runpath_exists(
    create_dummy_run_path,
    active_mask: list,
    expected: bool,
    use_tmpdir,
):
    def get_run_path_mock(realizations, iteration=None):
        if iteration is not None:
            return [f"out/realization-{r}/iter-{iteration}" for r in realizations]
        return [f"out/realization-{r}" for r in realizations]

    run_model = create_run_model(
        active_realizations=active_mask,
    )
    run_model._run_paths.get_paths = get_run_path_mock
    assert run_model.check_if_runpath_exists() == expected


def test_create_mask_from_failed_realizations_returns_initial_active_realizations_if_no_realization_succeeded(  # noqa
    use_tmpdir,
):
    initial_active_realizations = [True, False]
    active_realizations = initial_active_realizations.copy()
    completed_realizations = [False, False]

    brm = create_run_model(
        start_iteration=0,
        _total_iterations=1,
        active_realizations=active_realizations,
    )
    brm._initial_realizations_mask = initial_active_realizations
    brm.active_realizations = active_realizations
    brm._completed_realizations_mask = completed_realizations

    failed_realization_mask = brm._create_mask_from_failed_realizations()

    assert failed_realization_mask == initial_active_realizations


# TODO remove this test?
def test_run_model_logs_number_of_parameters(use_tmpdir):
    parameters = GenKwConfig(
        distribution={"name": "normal", "mean": 0, "std": 1},
        name="parameter_configuration",
        forward_init=False,
        update=True,
    )

    rm = create_run_model(parameter_configuration=[parameters])

    def mock_logging(_, log_str):
        regex = r"'num_parameters': (\d+)"
        match = re.search(regex, log_str)
        num_param = int(match.group(1))

        assert num_param == 1

    with patch.object(Logger, "info", mock_logging):
        rm.log_at_startup()


def test_that_defaulted_user_queue_options_overrides_site_queue_options(use_tmpdir):
    user_queue_options = LsfQueueOptions(
        max_running=0,
        submit_sleep=0,
        num_cpu=1,
        realization_memory=0,
    )

    class DummyValidationInfo:
        @property
        def context(self) -> ErtRuntimePlugins:
            return ErtRuntimePlugins(
                queue_options=LsfQueueOptions(
                    max_running=2, submit_sleep=2, num_cpu=2, realization_memory=2
                )
            )

    user_queue_config = QueueConfig(
        queue_system=QueueSystem.LSF, queue_options=user_queue_options
    )
    RunModel.inject_site_configuration_queue_options(
        user_queue_config, info=DummyValidationInfo()
    )

    assert user_queue_config.queue_options.max_running == 0
    assert user_queue_config.queue_options.submit_sleep == 0
    assert user_queue_config.queue_options.num_cpu == 1
    assert user_queue_config.queue_options.realization_memory == 0
