from ert.experiment_server import StateMachine


def generate_structure():
    return {
        "ee_1": {
            "ee_1/real_1": {"ee_1/real_1/step_1": ["ee_1/real_1/step_1/job_1"]},
            "ee_1/real_2": {"ee_1/real_2/step_1": ["ee_1/real_2/step_1/job_1"]},
        }
    }


def test_no_update():
    ensemble_structure = generate_structure()
    sm = StateMachine(ensemble_structure)
    assert sm.get_full_state == ensemble_structure


def test_single_update():
    ensemble_structure = generate_structure()
    sm = StateMachine(ensemble_structure)
    content = {"meta": "irrelevant_data"}
    sm.add_event({"ee_1/real_1/step_1/job_1": content})
    expected_update = {"ee_1/real_1/step_1/job_1": content}

    update_state = sm.get_update()
    assert update_state == expected_update


def test_multiple_updates():
    ensemble_structure = generate_structure()
    sm = StateMachine(ensemble_structure)

    content = {"meta": "irrelevant_data"}
    sm.add_event({"ee_1/real_1/step_1/job_1": content})
    sm.add_event({"ee_1/real_2/step_1/job_1": content})
    expected_update = {
        "ee_1/real_1/step_1/job_1": content,
        "ee_1/real_2/step_1/job_1": content,
    }

    update_state = sm.get_update()
    assert update_state == expected_update


def test_redundant_updates():
    ensemble_structure = generate_structure()
    sm = StateMachine(ensemble_structure)

    content = {"meta": "new_irrelevant_data"}
    sm.add_event({"ee_1/real_1/step_1/job_1": content})
    sm.add_event({"ee_1/real_1/step_1/job_1": content})
    expected_update = {"ee_1/real_1/step_1/job_1": content}
    update_state = sm.get_update()
    assert update_state == expected_update


def test_partly_reduntant():
    # Test that multiple events with duplicate content will be aggregated as expected
    ensemble_structure = generate_structure()
    sm = StateMachine(ensemble_structure)

    expected_content = {}
    # The identical content will be added to both events
    identical_content = {"meta": "irrelevant_data"}

    content = {"first_unique": 1}
    content.update(identical_content)
    sm.add_event({"ee_1/real_1/step_1/job_1": content})
    expected_content.update(content)

    content = {"second_unique": 1}
    content.update(identical_content)
    sm.add_event({"ee_1/real_1/step_1/job_1": content})
    expected_content.update(content)

    expected_update = {"ee_1/real_1/step_1/job_1": expected_content}

    update_state = sm.get_update()
    assert update_state == expected_update


def test_retrieve_only_new_information():
    ensemble_structure = generate_structure()
    sm = StateMachine(ensemble_structure)

    first_content = {"first": "irrelevant_data"}
    sm.add_event({"ee_1/real_1/step_1/job_1": first_content})

    expected_update = {"ee_1/real_1/step_1/job_1": first_content}

    # First update shall include the added event
    update_state = sm.get_update()
    assert update_state == expected_update

    second_content = {"second": "irrelevant_data"}
    sm.add_event({"ee_1/real_1/step_1/job_1": second_content})

    expected_update = {"ee_1/real_1/step_1/job_1": second_content}

    # Second update should only include events since last update
    update_state = sm.get_update()
    assert update_state == expected_update

    # A full state should include both contents
    content = first_content
    content.update(second_content)

    expected_state = ensemble_structure
    expected_state["ee_1"]["ee_1/real_1"]["ee_1/real_1/step_1"][
        "ee_1/real_1/step_1/job_1"
    ] = content

    assert sm.get_full_state == expected_state


def test_get_full_state_change_on_update():
    """The result of full state should only include events up to the last update.
    If events are added after that, the full state should not include those."""
    ensemble_structure = generate_structure()
    sm = StateMachine(ensemble_structure)

    first_content = {"first": "irrelevant_data"}
    sm.add_event({"ee_1/real_1/step_1/job_1": first_content})

    # Without a call to get_update we should receive empty state
    assert sm.get_full_state == ensemble_structure

    expected_update = {"ee_1/real_1/step_1/job_1": first_content}

    # First update shall include the added event
    update_state = sm.get_update()
    assert update_state == expected_update

    # And will now also be reflected in full_state
    expected_state["ee_1"]["ee_1/real_1"]["ee_1/real_1/step_1"][
        "ee_1/real_1/step_1/job_1"
    ] = first_content

    assert sm.get_full_state == expected_state

    second_content = {"second": "irrelevant_data"}
    sm.add_event({"ee_1/real_1/step_1/job_1": second_content})

    # Full state not affected yet
    assert sm.get_full_state == expected_state

    expected_update = {"ee_1/real_1/step_1/job_1": second_content}

    # Second update should only include events since last update
    update_state = sm.get_update()
    assert update_state == expected_update

    expected_state = ensemble_structure
    expected_state["ee_1"]["ee_1/real_1"]["ee_1/real_1/step_1"][
        "ee_1/real_1/step_1/job_1"
    ].update(second_content)

    # Full state should now reflect both contents
    assert sm.get_full_state == expected_state
