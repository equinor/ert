import pytest

from _ert.events import (
    EnsembleStarted,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    RealizationSuccess,
    RealizationWaiting,
)
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.snapshot import (
    EnsembleSnapshot,
    FMStepSnapshot,
    RealizationSnapshot,
)


@pytest.mark.parametrize(
    ("ensemble_size", "forward_models", "memory_reports"),
    [
        (10, 10, 1),
        (100, 10, 1),
        (10, 100, 1),
        (10, 10, 10),
    ],
)
def test_snapshot_handling_of_forward_model_events(
    benchmark, ensemble_size, forward_models, memory_reports
):
    benchmark(
        simulate_forward_model_event_handling,
        ensemble_size,
        forward_models,
        memory_reports,
    )


def simulate_forward_model_event_handling(
    ensemble_size, forward_models, memory_reports
):
    snapshot = EnsembleSnapshot()
    snapshot._ensemble_state = state.ENSEMBLE_STATE_UNKNOWN
    snapshot._metadata = {"foo": "bar"}

    for real in range(ensemble_size):
        realization = RealizationSnapshot(
            active=True, status=state.REALIZATION_STATE_WAITING, fm_steps={}
        )
        for fm_idx in range(forward_models):
            realization["fm_steps"][str(fm_idx)] = FMStepSnapshot(
                status=state.FORWARD_MODEL_STATE_START,
                index=str(fm_idx),
                name=f"FM_{fm_idx}",
            )
        snapshot.add_realization(str(real), realization)

    ens_id = "A"
    snapshot.update_from_event(EnsembleStarted(ensemble=ens_id))

    for real in range(ensemble_size):
        snapshot.update_from_event(RealizationWaiting(ensemble=ens_id, real=str(real)))

    for fm_idx in range(forward_models):
        for real in range(ensemble_size):
            snapshot.update_from_event(
                ForwardModelStepStart(
                    ensemble=ens_id,
                    real=str(real),
                    fm_step=str(fm_idx),
                    std_err="foo",
                    std_out="bar",
                )
            )
        for current_memory_usage in range(memory_reports):
            for real in range(ensemble_size):
                snapshot.update_from_event(
                    ForwardModelStepRunning(
                        ensemble=ens_id,
                        real=str(real),
                        fm_step=str(fm_idx),
                        max_memory_usage=current_memory_usage,
                        current_memory_usage=current_memory_usage,
                    )
                )
        for real in range(ensemble_size):
            snapshot.update_from_event(
                ForwardModelStepSuccess(
                    ensemble=ens_id, real=str(real), fm_step=str(fm_idx)
                ),
            )

    for real in range(ensemble_size):
        snapshot.update_from_event(RealizationSuccess(ensemble=ens_id, real=str(real)))
