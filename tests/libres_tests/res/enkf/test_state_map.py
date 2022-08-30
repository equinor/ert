import pytest

from ert._c_wrappers.enkf.enums.realization_state_enum import RealizationStateEnum
from ert._c_wrappers.enkf.state_map import StateMap


def test_state_map(tmp_path):
    state_map = StateMap()

    assert len(state_map) == 0

    with pytest.raises(TypeError):
        _ = state_map["r"]

    with pytest.raises(IOError):
        StateMap("DoesNotExist")

    with pytest.raises(IOError):
        state_map.load("/file/does/not/exist")

    _ = state_map[0]

    with pytest.raises(TypeError):
        state_map["r"] = RealizationStateEnum.STATE_INITIALIZED

    with pytest.raises(TypeError):
        state_map[0] = "INITIALIZED"

    with pytest.raises(IndexError):
        state_map[-1] = RealizationStateEnum.STATE_INITIALIZED

    state_map[0] = RealizationStateEnum.STATE_INITIALIZED

    assert len(state_map) == 1

    state_map[1] = RealizationStateEnum.STATE_INITIALIZED
    state_map[1] = RealizationStateEnum.STATE_HAS_DATA

    assert len(state_map) == 2

    index = 0
    for state in state_map:
        assert state == state_map[index]
        index += 1

    states = list(state_map)

    assert states == [
        RealizationStateEnum.STATE_INITIALIZED,
        RealizationStateEnum.STATE_HAS_DATA,
    ]

    state_map[5] = RealizationStateEnum.STATE_INITIALIZED
    assert len(state_map) == 6

    assert state_map[2] == RealizationStateEnum.STATE_UNDEFINED
    assert state_map[3] == RealizationStateEnum.STATE_UNDEFINED
    assert state_map[4] == RealizationStateEnum.STATE_UNDEFINED
    assert state_map[5] == RealizationStateEnum.STATE_INITIALIZED

    state_map.save(str(tmp_path / "MAP"))
    assert state_map == StateMap(str(tmp_path / "MAP"))


def test_state_map_transitions():
    assert StateMap.isLegalTransition(
        RealizationStateEnum.STATE_UNDEFINED,
        RealizationStateEnum.STATE_INITIALIZED,
    )
    assert StateMap.isLegalTransition(
        RealizationStateEnum.STATE_INITIALIZED,
        RealizationStateEnum.STATE_HAS_DATA,
    )
    assert StateMap.isLegalTransition(
        RealizationStateEnum.STATE_INITIALIZED,
        RealizationStateEnum.STATE_LOAD_FAILURE,
    )
    assert StateMap.isLegalTransition(
        RealizationStateEnum.STATE_INITIALIZED,
        RealizationStateEnum.STATE_PARENT_FAILURE,
    )
    assert StateMap.isLegalTransition(
        RealizationStateEnum.STATE_HAS_DATA,
        RealizationStateEnum.STATE_PARENT_FAILURE,
    )

    assert not (
        StateMap.isLegalTransition(
            RealizationStateEnum.STATE_UNDEFINED,
            RealizationStateEnum.STATE_LOAD_FAILURE,
        )
    )
    assert not (
        StateMap.isLegalTransition(
            RealizationStateEnum.STATE_UNDEFINED,
            RealizationStateEnum.STATE_HAS_DATA,
        )
    )

    with pytest.raises(TypeError):
        StateMap.isLegalTransition("error", RealizationStateEnum.STATE_UNDEFINED)

    with pytest.raises(TypeError):
        StateMap.isLegalTransition(RealizationStateEnum.STATE_UNDEFINED, "error")

    with pytest.raises(TypeError):
        StateMap.isLegalTransition("error", "exception")


def test_active_list():
    state_map = StateMap()
    state_map[0] = RealizationStateEnum.STATE_INITIALIZED
    state_map[2] = RealizationStateEnum.STATE_INITIALIZED
    state_map[2] = RealizationStateEnum.STATE_HAS_DATA

    initialized = state_map.realizationList(RealizationStateEnum.STATE_INITIALIZED)
    assert len(initialized) == 1
    assert initialized[0] == 0

    mask = state_map.createMask(RealizationStateEnum.STATE_INITIALIZED)
    assert list(mask) == [True, False, False]

    has_data = state_map.realizationList(RealizationStateEnum.STATE_HAS_DATA)
    assert len(has_data) == 1
    assert has_data[0] == 2

    mask = state_map.createMask(RealizationStateEnum.STATE_HAS_DATA)
    assert list(mask) == [False, False, True]

    state = RealizationStateEnum.STATE_HAS_DATA | RealizationStateEnum.STATE_INITIALIZED
    mask = state_map.createMask(state)
    assert list(mask) == [True, False, True]
