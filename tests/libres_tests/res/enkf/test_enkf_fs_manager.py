import sys

import pytest

from ert._c_wrappers.enkf import RunContext


def test_enkf_fs_manager_create(snake_oil_case):
    # We are indirectly testing the create through the create
    # already in the enkf_main object. In principle we could
    # create a separate manager instance from the ground up, but
    # then the reference count will be weird.
    ert = snake_oil_case
    fsm = ert.getEnkfFsManager()

    fsm.getCurrentFileSystem()
    assert fsm.caseExists("default_0")
    assert fsm.caseHasData("default_0")

    assert not fsm.caseExists("newFS")
    assert not fsm.caseHasData("newFS")

    fsm.getFileSystem("newFS")

    assert fsm.caseExists("newFS")
    assert not fsm.caseHasData("newFS")


def test_rotate(snake_oil_case):
    ert = snake_oil_case
    fsm = ert.getEnkfFsManager()
    assert len(fsm._fs_rotator) == 2

    fs_list = []
    for index in range(5):
        fs_list.append(fsm.getFileSystem(f"fs_fill_{index}"))

    assert len(fsm._fs_rotator) == 7

    for index in range(3 * 5):
        fs_name = f"fs_test_{index}"
        sys.stderr.write(f"Mounting: {fs_name}\n")
        fsm.getFileSystem(fs_name)
        assert len(fsm._fs_rotator) == 8 + index


@pytest.mark.parametrize(
    "state_mask, expected_length",
    [([True] * 25, 25), ([False] * 25, 25), ([False, True, True], 25)],
)
def test_custom_init_runs(snake_oil_case, state_mask, expected_length):
    ert = snake_oil_case
    new_fs = ert.getEnkfFsManager().getFileSystem("new_case")
    ert.getEnkfFsManager().switchFileSystem(new_fs)
    ert.getEnkfFsManager().customInitializeCurrentFromExistingCase(
        "default_0", 0, state_mask, ["SNAKE_OIL_PARAM"]
    )
    assert len(ert.getEnkfFsManager().getStateMapForCase("new_case")) == expected_length


def test_fs_init_from_scratch(snake_oil_case):
    ert = snake_oil_case
    sim_fs = ert.getEnkfFsManager().getFileSystem("new_case")
    mask = [True] * 6 + [False] * 19
    run_context = RunContext(sim_fs=sim_fs, mask=mask)

    ert.getEnkfFsManager().initRun(run_context, ["SNAKE_OIL_PARAM"])
    assert len(ert.getEnkfFsManager().getStateMapForCase("new_case")) == 25
