import sys

import pytest

from ert._c_wrappers.enkf import RunContext


def test_enkf_fs_manager_create(snake_oil_case):
    # We are indirectly testing the create through the create
    # already in the enkf_main object. In principle we could
    # create a separate manager instance from the ground up, but
    # then the reference count will be weird.
    ert = snake_oil_case
    fsm = ert.storage_manager

    assert "default_0" in fsm
    assert fsm.has_data("default_0")

    assert "newFS" not in fsm

    fsm.add_case("newFS")

    assert "newFS" in fsm
    assert not fsm.has_data("newFS")


def test_rotate(snake_oil_case):
    ert = snake_oil_case
    fsm = ert.getEnkfFsManager()
    assert len(fsm.storage_manager) == 2

    fs_list = []
    for index in range(5):
        fs_list.append(fsm.getFileSystem(f"fs_fill_{index}"))

    assert len(fsm.storage_manager) == 7

    for index in range(3 * 5):
        fs_name = f"fs_test_{index}"
        sys.stderr.write(f"Mounting: {fs_name}\n")
        fsm.getFileSystem(fs_name)
        assert len(fsm.storage_manager) == 8 + index


@pytest.mark.parametrize(
    "state_mask, expected_length",
    [([True] * 25, 25), ([False] * 25, 25), ([False, True, True], 25)],
)
def test_custom_init_runs(snake_oil_case, state_mask, expected_length):
    ert = snake_oil_case
    fs_manager = ert.storage_manager
    source_fs = fs_manager.current_case
    new_fs = fs_manager.add_case("new_case")
    source_fs.copy_from_case(new_fs, 0, ["SNAKE_OIL_PARAM"], state_mask)
    assert len(new_fs.getStateMap()) == expected_length


def test_fs_init_from_scratch(snake_oil_case):
    ert = snake_oil_case
    sim_fs = ert.getEnkfFsManager().getFileSystem("new_case")
    mask = [True] * 6 + [False] * 19
    run_context = RunContext(sim_fs=sim_fs, mask=mask)

    ert.getEnkfFsManager().sample_prior(
        run_context.sim_fs, run_context.active_realizations, ["SNAKE_OIL_PARAM"]
    )
    assert len(ert.getEnkfFsManager().getStateMapForCase("new_case")) == 25
