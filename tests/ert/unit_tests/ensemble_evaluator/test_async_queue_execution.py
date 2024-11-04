import pytest

from ert.scheduler import Scheduler, create_driver


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_happy_path(
    tmpdir,
    make_ensemble,
    queue_config,
    monkeypatch,
):
    ensemble = make_ensemble(monkeypatch, tmpdir, 1, 1)

    queue = Scheduler(
        driver=create_driver(queue_config.queue_options),
        realizations=ensemble.reals,
        ens_id="ee_0",
    )

    await queue.execute()

    type_states = ["waiting", "waiting", "pending", "running", "success"]
    event_states = ["WAITING", "SUBMITTING", "PENDING", "RUNNING", "COMPLETED"]
    id_state = 0
    while not queue._events.empty():
        event = await queue._events.get()
        assert event.ensemble == "ee_0"
        assert event.real == "0"
        assert event.event_type == f"realization.{type_states[id_state]}"
        assert event.queue_event_type == event_states[id_state]
        id_state += 1
