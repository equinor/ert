from ert.ensemble_evaluator import state
from ert.gui.experiments.view import run_status as run_status_module
from ert.gui.experiments.view.run_status import RunStatusView
from ert.run_models.event import FullSnapshotEvent
from tests.ert.utils import SnapshotBuilder


def _build_run_status_snapshot(iteration: int) -> FullSnapshotEvent:
    return FullSnapshotEvent(
        snapshot=SnapshotBuilder()
        .add_fm_step(
            fm_step_id="0",
            index="0",
            name="fm_step_0",
            status=state.FORWARD_MODEL_STATE_FINISHED,
        )
        .build(
            real_ids=["0", "1"],
            status=state.REALIZATION_STATE_FINISHED,
        ),
        iteration_label=f"Running forecast for iteration: {iteration}",
        total_iterations=1,
        progress=1.0,
        realization_count=2,
        status_count={"Finished": 2},
        iteration=iteration,
    )


def test_that_run_status_view_reuses_loaded_snapshot_when_path_is_unchanged(
    qtbot, tmp_path, monkeypatch
):
    path = tmp_path / "snapshot_0.json"
    load_call_count = 0

    def load_snapshot(_):
        nonlocal load_call_count
        load_call_count += 1
        return _build_run_status_snapshot(iteration=0)

    monkeypatch.setattr(run_status_module, "load_status_snapshot_event", load_snapshot)

    run_status_view = RunStatusView()
    qtbot.addWidget(run_status_view)

    run_status_view.load_snapshot(path)
    assert load_call_count == 1
    content_after_first_load = run_status_view._content

    run_status_view.load_snapshot(path)

    assert load_call_count == 1
    assert run_status_view._content is content_after_first_load
