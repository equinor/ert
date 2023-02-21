import shutil

import pytest

import ert
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert._c_wrappers.enkf.enkf_fs_manager import FS_VERSION, FileSystemError


def test_enkf_fs_manager_create(snake_oil_case_storage):
    # We are indirectly testing the create through the create
    # already in the enkf_main object. In principle we could
    # create a separate manager instance from the ground up, but
    # then the reference count will be weird.
    ert = snake_oil_case_storage
    fsm = ert.storage_manager

    assert "default_0" in fsm
    assert fsm.has_data("default_0")

    assert "newFS" not in fsm

    fsm.add_case("newFS")

    assert "newFS" in fsm
    assert not fsm.has_data("newFS")


@pytest.mark.parametrize(
    "current_fs_version, expected_error",
    [
        (FS_VERSION + 1, "created by an older"),
        (FS_VERSION - 1, "created by a newer"),
    ],
)
def test_enkf_fs_manager_wrong_version(
    copy_snake_oil_case_storage, monkeypatch, current_fs_version, expected_error
):
    monkeypatch.setattr(
        ert._c_wrappers.enkf.enkf_fs_manager, "FS_VERSION", current_fs_version
    )

    with pytest.raises(FileSystemError) as e:
        EnKFMain(ErtConfig.from_file("snake_oil.ert"))
    assert expected_error in str(e.value)


def test_rotate(snake_oil_case):
    ert = snake_oil_case
    fsm = ert.storage_manager
    assert len(fsm) == 1

    fs_list = []
    for index in range(5):
        fs_list.append(fsm.add_case(f"fs_fill_{index}"))

    assert len(fsm) == 6

    for index in range(3 * 5):
        fs_name = f"fs_test_{index}"
        fsm.add_case(fs_name)
        assert len(fsm) == 7 + index
        assert len(fsm.open_storages) == 5


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
    mask = [True] * 6 + [False] * 19
    run_context = ert.create_ensemble_context("new_case", mask, 0)

    ert.sample_prior(
        run_context.sim_fs,
        run_context.active_realizations,
        ["SNAKE_OIL_PARAM"],
    )
    assert len(ert.storage_manager.state_map("new_case")) == 25


def test_missing_current_case(snake_oil_case):
    ert = snake_oil_case
    current_case_name = ert.storage_manager.active_case
    config_file = ert.ert_config.user_config_file
    storage_path = ert.storage_manager.storage_path

    shutil.rmtree(storage_path / current_case_name)
    ert = EnKFMain(ErtConfig.from_file(config_file))
    assert ert.storage_manager.active_case == current_case_name
