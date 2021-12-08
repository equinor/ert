import shutil
import os
import pytest
from res.enkf import ResConfig, EnKFMain, ErtRunContext
from ecl.util.util import BoolVector, StringList
from ert_shared import __version__
from packaging import version


@pytest.fixture()
def copy_snake_oil(tmpdir, source_root):
    test_data_dir = os.path.join(source_root, "test-data", "local", "snake_oil")
    with tmpdir.as_cwd():
        shutil.copytree(test_data_dir, "test-data")
        os.chdir("test-data")
        yield


@pytest.fixture()
def fail_if_not_removed():
    if version.parse(__version__) > version.parse("2.40"):
        pytest.fail(
            (
                "This has passed its deprecation period "
                "and the corresponding argument should be removed"
            )
        )


@pytest.mark.usefixtures("copy_snake_oil")
@pytest.mark.parametrize(
    "state_mask, expected_length",
    [([True] * 25, 25), ([False] * 25, 0), ([False, True, True], 3)],
)
def test_custom_init_runs(state_mask, expected_length):
    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)
    new_fs = ert.getEnkfFsManager().getFileSystem("new_case")  # new case
    ert.getEnkfFsManager().switchFileSystem(new_fs)
    index_list = [i for i, flag in enumerate(state_mask) if flag]
    bool_vector = BoolVector.createFromList(len(state_mask), index_list)
    ert.getEnkfFsManager().customInitializeCurrentFromExistingCase(
        "default_0", 0, bool_vector, StringList(["SNAKE_OIL_PARAM"])
    )
    assert len(ert.getEnkfFsManager().getStateMapForCase("new_case")) == expected_length


@pytest.mark.usefixtures("copy_snake_oil")
def test_fs_init_from_scratch():
    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)
    sim_fs = ert.getEnkfFsManager().getFileSystem("new_case")  # new case
    mask = BoolVector.createFromList(25, [0, 1, 2, 3, 4, 5])
    run_context = ErtRunContext.case_init(sim_fs, mask)

    ert.getEnkfFsManager().initializeFromScratch(
        StringList(["SNAKE_OIL_PARAM"]), run_context
    )
    assert len(ert.getEnkfFsManager().getStateMapForCase("new_case")) == 6


@pytest.mark.usefixtures("copy_snake_oil", "fail_if_not_removed")
@pytest.mark.parametrize(
    "state_mask, expected_length",
    [([True] * 25, 25), ([False] * 25, 0), ([False, True, True], 3)],
)
def test_custom_init_runs_deprecated(state_mask, expected_length):
    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)
    new_fs = ert.getEnkfFsManager().getFileSystem("new_case")  # new case
    ert.getEnkfFsManager().switchFileSystem(new_fs)
    index_list = [i for i, flag in enumerate(state_mask) if flag]
    bool_vector = BoolVector.createFromList(len(state_mask), index_list)
    with pytest.warns(DeprecationWarning):
        ert.getEnkfFsManager().customInitializeCurrentFromExistingCase(
            "default_0", 0, bool_vector, StringList(["SNAKE_OIL_PARAM"])
        )
    assert len(ert.getEnkfFsManager().getStateMapForCase("new_case")) == expected_length


@pytest.mark.usefixtures("copy_snake_oil", "fail_if_not_removed")
def test_fs_init_from_scratch_deprecated():
    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)
    sim_fs = ert.getEnkfFsManager().getFileSystem("new_case")  # new case
    mask = BoolVector.createFromList(25, [0, 1, 2, 3, 4, 5])
    run_context = ErtRunContext.case_init(sim_fs, mask)
    with pytest.warns(DeprecationWarning):
        ert.getEnkfFsManager().initializeFromScratch(
            StringList(["SNAKE_OIL_PARAM"]), run_context
        )
    assert len(ert.getEnkfFsManager().getStateMapForCase("new_case")) == 6
