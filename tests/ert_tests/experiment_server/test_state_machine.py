from cloudevents.http import CloudEvent
from collections import defaultdict
import copy

from ert.experiment_server import StateMachine
from ert.experiment_server._state_machine import nested_dict_keys
from ert.ensemble_evaluator import identifiers as ids


def _generate_structure():
    return {
        "ee_1": {
            "ee_1/real_1": {"ee_1/real_1/step_1": ["ee_1/real_1/step_1/job_1"]},
            "ee_1/real_2": {"ee_1/real_2/step_1": ["ee_1/real_2/step_1/job_1"]},
        }
    }


def _empty_state_from_structure(structure):
    return defaultdict(dict, {k: {} for k in nested_dict_keys(structure)})


def _update_from_cloudevent(event):
    return {event["source"]: {"type": event["type"], **event.data}}


def test_no_update():
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)
    assert sm.get_full_state() == _empty_state_from_structure(ensemble_structure)


def test_single_update():
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)
    event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee_1/real_1/step_1/job_1",
        },
        data={"meta": "irrelevant_data"},
    )
    sm.queue_event(event)
    sm.apply_updates()
    assert sm.get_update() == _update_from_cloudevent(event)

    state = _empty_state_from_structure(ensemble_structure)
    state["ee_1/real_1/step_1/job_1"] = {
        "type": ids.EVTYPE_FM_JOB_RUNNING,
        "meta": "irrelevant_data",
    }
    assert sm.get_full_state() == state


def test_multiple_updates():
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)

    content = {"meta": "irrelevant_data"}
    first_event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee_1/real_1/step_1/job_1",
        },
        data=content,
    )
    second_event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee_1/real_2/step_1/job_1",
        },
        data=content,
    )
    sm.queue_event(first_event)
    sm.queue_event(second_event)

    expected_update = {}
    expected_update.update(_update_from_cloudevent(first_event))
    expected_update.update(_update_from_cloudevent(second_event))

    sm.apply_updates()
    assert sm.get_update() == expected_update

    expected_state = _empty_state_from_structure(ensemble_structure)

    expected_state["ee_1/real_1/step_1/job_1"] = {
        "type": ids.EVTYPE_FM_JOB_RUNNING,
        **content,
    }
    expected_state["ee_1/real_2/step_1/job_1"] = {
        "type": ids.EVTYPE_FM_JOB_RUNNING,
        **content,
    }

    assert sm.get_full_state() == expected_state


def test_redundant_updates():
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)

    event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee_1/real_1/step_1/job_1",
        },
        data={"meta": "irrelevant_data"},
    )
    sm.queue_event(event)

    event.data = {"meta": "new_irrelevant_data"}
    sm.queue_event(event)

    sm.apply_updates()

    assert sm.get_update() == _update_from_cloudevent(event)


def test_partly_reduntant():
    # Test that multiple events with duplicate content will be aggregated as expected
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)

    event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee_1/real_1/step_1/job_1",
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

    sm.queue_event(event)

    event = copy.copy(event)
    content = {"second_unique": 2}
    content.update(identical_content)
    expected_content.update(content)
    event.data = content

    sm.queue_event(event)

    sm.apply_updates()
    update_state = sm.get_update()
    assert update_state["ee_1/real_1/step_1/job_1"] == expected_content


def test_retrieve_only_new_information():
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)

    event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee_1/real_1/step_1/job_1",
        },
    )
    event.data = {"some": "irrelevant_data"}

    sm.queue_event(event)
    sm.apply_updates()

    # First update shall include the added event
    assert sm.get_update() == _update_from_cloudevent(event)

    event = copy.copy(event)
    event.data = {"another": "irrelevant_data"}
    sm.queue_event(event)
    sm.apply_updates()

    # Second update should only include events since last update
    assert sm.get_update() == _update_from_cloudevent(event)

    # A full state should include both contents
    expected_state = _empty_state_from_structure(ensemble_structure)
    expected_state["ee_1/real_1/step_1/job_1"] = {
        "type": ids.EVTYPE_FM_JOB_RUNNING,
        "some": "irrelevant_data",
        "another": "irrelevant_data",
    }

    assert sm.get_full_state() == expected_state


def test_get_full_state_change_on_update():
    """The result of full state should only include events up to the last update.
    If events are added after that, the full state should not include those until apply_updates is called"""
    ensemble_structure = _generate_structure()
    sm = StateMachine(ensemble_structure)

    event = CloudEvent(
        {
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "source": "ee_1/real_1/step_1/job_1",
        },
    )
    event.data = {"some": "irrelevant_data"}

    sm.queue_event(event)

    # Without a call to apply_updates we should receive empty state
    expected_state = _empty_state_from_structure(ensemble_structure)
    assert sm.get_full_state() == expected_state

    sm.apply_updates()

    # First update shall include the added event
    assert sm.get_update() == _update_from_cloudevent(event)

    # And will now also be reflected in full_state
    expected_state["ee_1/real_1/step_1/job_1"] = {
        "type": ids.EVTYPE_FM_JOB_RUNNING,
        "some": "irrelevant_data",
    }
    assert sm.get_full_state() == expected_state

    event = copy.copy(event)
    event.data = {"another": "irrelevant_data"}
    sm.queue_event(event)

    # Full state not affected yet
    assert sm.get_full_state() == expected_state

    sm.apply_updates()

    # Second update should only include events since last update
    assert sm.get_update() == _update_from_cloudevent(event)

    # A full state should now reflect both contents
    expected_state["ee_1/real_1/step_1/job_1"]["another"] = "irrelevant_data"

    assert sm.get_full_state() == expected_state


def test_correct_presedence_misaligned_events():
    # Consider that one actor says things are good, another says it's not
    pass


def test_disallowed_state_change():
    # Going from failed to successfull, should it be allowed?
    pass


def test_large_case():
    # 500 realisations, 50 jobs
    pass


def test_propagate_state_change():
    # all jobs succeed -> step succeed
    pass
