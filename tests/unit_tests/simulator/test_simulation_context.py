import pytest

from ert import JobStatus
from ert.simulator import BatchContext
from tests.utils import wait_until


@pytest.mark.usefixtures("using_scheduler")
def test_simulation_context(setup_case, storage):
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
            assert even_ctx.job_status(iens) != JobStatus.SUCCESS
        else:
            assert odd_ctx.job_status(iens) != JobStatus.SUCCESS

    wait_until(lambda: not even_ctx.running() and not odd_ctx.running(), timeout=90)

    for iens in range(size):
        if iens % 2 == 0:
            assert even_ctx._run_context[iens].runpath.endswith(
                f"runpath/realization-{iens}-{iens}/iter-0"
            )
        else:
            assert odd_ctx._run_context[iens].runpath.endswith(
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
            assert even_ctx.job_status(iens) != JobStatus.FAILED
            assert even_ctx.job_status(iens) == JobStatus.SUCCESS
        else:
            assert odd_ctx.job_status(iens) != JobStatus.FAILED
            assert odd_ctx.job_status(iens) == JobStatus.SUCCESS
