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
        driver=create_driver(queue_config),
        realizations=ensemble.reals,
        ens_id="ee_0",
    )

    await queue.execute()

    type_states = ["waiting", "waiting", "pending", "running", "success"]
    event_states = ["WAITING", "SUBMITTING", "PENDING", "RUNNING", "COMPLETED"]
    id_state = 0
    while not queue._events.empty():
        received_event = await queue._events.get()
        assert received_event["source"] == "/ert/ensemble/ee_0/real/0"
        assert (
            received_event["type"]
            == f"com.equinor.ert.realization.{type_states[id_state]}"
        )
        assert received_event.data == {"queue_event_type": event_states[id_state]}
        id_state += 1
