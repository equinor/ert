import pytest

from ert import JobState, JobStatus
from ert.simulator import BatchContext
from tests.ert.utils import wait_until


@pytest.mark.parametrize(
    "success_state, failure_state, status_check_method_name",
    [
        pytest.param(
            JobState.COMPLETED, JobState.FAILED, "get_job_state", id="current"
        ),
        pytest.param(JobStatus.SUCCESS, JobStatus.FAILED, "job_status", id="legacy"),
    ],
)
def test_simulation_context(
    success_state, failure_state, status_check_method_name, setup_case, storage
):
    ert_config = setup_case("batch_sim", "sleepy_time.ert")

    size = 4
    even_mask = [True, False] * (size // 2)
    odd_mask = [False, True] * (size // 2)

    experiment_id = storage.create_experiment()
    even_half = storage.create_ensemble(
        experiment_id,
        name="even_half",
        ensemble_size=ert_config.model_config.num_realizations,
    )
    odd_half = storage.create_ensemble(
        experiment_id,
        name="odd_half",
        ensemble_size=ert_config.model_config.num_realizations,
    )

    case_data = [(geo_id, {}) for geo_id in range(size)]
    even_ctx = BatchContext([], ert_config, even_half, even_mask, 0, case_data)
    odd_ctx = BatchContext([], ert_config, odd_half, odd_mask, 0, case_data)

    for iens in range(size):
        if iens % 2 == 0:
            assert getattr(even_ctx, status_check_method_name)(iens) != success_state
        else:
            assert getattr(odd_ctx, status_check_method_name)(iens) != success_state

    wait_until(lambda: not even_ctx.running() and not odd_ctx.running(), timeout=90)

    for iens in range(size):
        if iens % 2 == 0:
            assert even_ctx.run_args[iens].runpath.endswith(
                f"runpath/realization-{iens}-{iens}/iter-0"
            )
        else:
            assert odd_ctx.run_args[iens].runpath.endswith(
                f"runpath/realization-{iens}-{iens}/iter-0"
            )

    assert even_ctx.status.failed == 0
    assert even_ctx.status.running == 0
    assert even_ctx.status.complete == size / 2

    assert odd_ctx.status.failed == 0
    assert odd_ctx.status.running == 0
    assert odd_ctx.status.complete == size / 2

    for iens in range(size):
        if iens % 2 == 0:
            assert getattr(even_ctx, status_check_method_name)(iens) != failure_state
            assert getattr(even_ctx, status_check_method_name)(iens) == success_state
        else:
            assert getattr(odd_ctx, status_check_method_name)(iens) != failure_state
            assert getattr(odd_ctx, status_check_method_name)(iens) == success_state
