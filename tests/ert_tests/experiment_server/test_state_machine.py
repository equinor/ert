import asyncio
from cloudevents.http import CloudEvent
from collections import defaultdict
import copy
import itertools
import pytest

from ert.experiment_server import StateMachine
from ert.experiment_server._state_machine import nested_dict_keys
from ert.ensemble_evaluator import identifiers as ids


def _generate_structure(n_reals=2, n_steps=2, n_jobs=2):
    return {
        "ee/0": {
            f"ee/0/real/{i}": {
                f"ee/0/real/{i}/step/{j}": [
                    f"ee/0/real/{i}/step/{j}/job/{k}" for k in range(n_jobs)
                ]
                for j in range(n_steps)
            }
            for i in range(n_reals)
        }
    }


def _empty_state_from_structure(structure):
    return defaultdict(dict, {k: {} for k in nested_dict_keys(structure)})


def _update_from_cloudevent(event):
    return {event["source"]: {"type": event["type"], **event.data}}


@pytest.mark.asyncio
async def test_no_update():
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)
    assert await sm.get_full_state() == _empty_state_from_structure(ensemble_structure)


@pytest.mark.asyncio
async def test_single_update():
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)
    event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee/0/real/0/step/0/job/0",
        },
        data={"meta": "irrelevant_data"},
    )
    await sm.queue_event(event)
    await sm.apply_updates()
    assert await sm.get_update() == _update_from_cloudevent(event)

    state = _empty_state_from_structure(ensemble_structure)
    state["ee/0/real/0/step/0/job/0"] = {
        "type": ids.EVTYPE_FM_JOB_RUNNING,
        "meta": "irrelevant_data",
    }
    assert await sm.get_full_state() == state


@pytest.mark.asyncio
async def test_multiple_updates():
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)

    content = {"meta": "irrelevant_data"}
    first_event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee/0/real/0/step/0/job/0",
        },
        data=content,
    )
    second_event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee/0/real/1/step/0/job/0",
        },
        data=content,
    )
    await sm.queue_event(first_event)
    await sm.queue_event(second_event)

    expected_update = {}
    expected_update.update(_update_from_cloudevent(first_event))
    expected_update.update(_update_from_cloudevent(second_event))

    await sm.apply_updates()
    assert await sm.get_update() == expected_update

    expected_state = _empty_state_from_structure(ensemble_structure)

    expected_state["ee/0/real/0/step/0/job/0"] = {
        "type": ids.EVTYPE_FM_JOB_RUNNING,
        **content,
    }
    expected_state["ee/0/real/1/step/0/job/0"] = {
        "type": ids.EVTYPE_FM_JOB_RUNNING,
        **content,
    }

    assert await sm.get_full_state() == expected_state


@pytest.mark.asyncio
async def test_redundant_updates():
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)

    event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee/0/real/0/step/0/job/0",
        },
        data={"meta": "irrelevant_data"},
    )
    await sm.queue_event(event)

    event.data = {"meta": "new_irrelevant_data"}
    await sm.queue_event(event)

    await sm.apply_updates()

    assert await sm.get_update() == _update_from_cloudevent(event)


@pytest.mark.asyncio
async def test_partly_reduntant():
    # Test that multiple events with duplicate content will be aggregated as expected
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)

    event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee/0/real/0/step/0/job/0",
        },
    )
    expected_content = {"type": ids.EVTYPE_FM_JOB_RUNNING}

    # The identical content will be added to both events
    identical_content = {"meta": "irrelevant_data"}
    expected_content.update(identical_content)

    content = {"first_unique": 1}
    content.update(identical_content)
    expected_content.update(content)
    event.data = content

    await sm.queue_event(event)

    event = copy.copy(event)
    content = {"second_unique": 2}
    content.update(identical_content)
    expected_content.update(content)
    event.data = content

    await sm.queue_event(event)

    await sm.apply_updates()
    update_state = await sm.get_update()
    assert update_state["ee/0/real/0/step/0/job/0"] == expected_content


@pytest.mark.asyncio
async def test_retrieve_only_new_information():
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)

    event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee/0/real/0/step/0/job/0",
        },
    )
    event.data = {"some": "irrelevant_data"}

    await sm.queue_event(event)
    await sm.apply_updates()

    # First update shall include the added event
    assert await sm.get_update() == _update_from_cloudevent(event)

    event = copy.copy(event)
    event.data = {"another": "irrelevant_data"}
    await sm.queue_event(event)
    await sm.apply_updates()

    # Second update should only include events since last update
    assert await sm.get_update() == _update_from_cloudevent(event)

    # A full state should include both contents
    expected_state = _empty_state_from_structure(ensemble_structure)
    expected_state["ee/0/real/0/step/0/job/0"] = {
        "type": ids.EVTYPE_FM_JOB_RUNNING,
        "some": "irrelevant_data",
        "another": "irrelevant_data",
    }

    assert await sm.get_full_state() == expected_state


@pytest.mark.asyncio
async def test_get_full_state_change_on_update():
    """The result of full state should only include events up to the last update.
    If events are added after that, the full state should not include those until apply_updates is called"""
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)

    event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee/0/real/0/step/0/job/0",
        },
    )
    event.data = {"some": "irrelevant_data"}

    await sm.queue_event(event)

    # Without a call to apply_updates we should receive empty state
    expected_state = _empty_state_from_structure(ensemble_structure)
    assert await sm.get_full_state() == expected_state

    await sm.apply_updates()

    # First update shall include the added event
    assert await sm.get_update() == _update_from_cloudevent(event)

    # And will now also be reflected in full_state
    expected_state["ee/0/real/0/step/0/job/0"] = {
        "type": ids.EVTYPE_FM_JOB_RUNNING,
        "some": "irrelevant_data",
    }
    assert await sm.get_full_state() == expected_state

    event = copy.copy(event)
    event.data = {"another": "irrelevant_data"}
    await sm.queue_event(event)

    # Full state not affected yet
    assert await sm.get_full_state() == expected_state

    await sm.apply_updates()

    # Second update should only include events since last update
    assert await sm.get_update() == _update_from_cloudevent(event)

    # A full state should now reflect both contents
    expected_state["ee/0/real/0/step/0/job/0"]["another"] = "irrelevant_data"

    assert await sm.get_full_state() == expected_state


@pytest.mark.asyncio
async def test_event_consumed_and_produced():
    ensemble_structure = _generate_structure(n_reals=1, n_steps=1, n_jobs=100)
    sm = StateMachine(ensemble_structure)

    for i in range(20):
        event = CloudEvent(
            {
                "type": ids.EVTYPE_FM_JOB_SUCCESS,
                "source": f"ee/0/real/0/step/0/job/{i}",
            },
            data={"meta": "irrelevant_data"},
        )
        await sm.queue_event(event)
        await asyncio.sleep(0.1)

    await sm.stop()


@pytest.mark.asyncio
async def test_propagate_state_change():
    """Test that derived transitions are handled correctly. I.e. given that events
    determining all steps in a realisation finished, the realisation itself should be
    set to finished
    """
    ensemble_structure = _generate_structure(n_reals=1, n_steps=1, n_jobs=1)
    sm = StateMachine(ensemble_structure)

    event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_SUCCESS,
            "source": "ee/0/real/0/step/0/job/0",
        },
    )

    await sm.queue_event(event)
    await sm.apply_updates()
    # all jobs succeed -> step succeed
    pass


def test_correct_presedence_misaligned_events():
    # Consider that one actor says things are good, another says it's not
    pass


def test_disallowed_state_change():
    # Going from failed to successfull, should it be allowed?
    pass


@pytest.mark.asyncio
async def test_large_case():
    n_reals = 500
    n_jobs = 100
    ensemble_structure = _generate_structure(n_reals=n_reals, n_jobs=n_jobs)
    sm = StateMachine(ensemble_structure)
    for i, j in itertools.product(range(n_reals), range(n_jobs)):
        await sm.queue_event(
            CloudEvent(
                {
                    "type": ids.EVTYPE_FM_JOB_START,
                    "source": f"ee/0/real/{i}/step/0/job/{j}",
                },
                data={"some": "irrelevant_data"},
            )
        )
        await sm.queue_event(
            CloudEvent(
                {
                    "type": ids.EVTYPE_FM_JOB_RUNNING,
                    "source": f"ee/0/real/{i}/step/0/job/{j}",
                },
                data={"some": "irrelevant_data"},
            )
        )
        await sm.queue_event(
            CloudEvent(
                {
                    "type": ids.EVTYPE_FM_JOB_SUCCESS,
                    "source": f"ee/0/real/{i}/step/0/job/{j}",
                },
                data={"some": "irrelevant_data"},
            )
        )

    import time

    start = time.time()
    await sm.apply_updates()
    duration = time.time() - start
    a = 2
