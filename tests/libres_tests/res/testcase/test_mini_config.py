from ert._c_wrappers.enkf import EnKFMain
from ert._c_wrappers.enkf.enums.realization_state_enum import RealizationStateEnum


def test_failed_realizations(setup_case):

    # mini_fail_config has the following realization success/failures:
    #
    # 0 OK
    # 1 GenData report step 1 missing
    # 2 GenData report step 2 missing, Forward Model Component Target File not found
    # 3 GenData report step 3 missing, Forward Model Component Target File not found
    # 4 GenData report step 1 missing
    # 5 GenData report step 2 missing, Forward Model Component Target File not found
    # 6 GenData report step 3 missing
    # 7 Forward Model Target File not found.
    # 8 OK
    # 9 OK
    ert = EnKFMain(setup_case("local/mini_ert", "mini_fail_config"))
    fs = ert.getEnkfFsManager().getCurrentFileSystem()

    realizations_list = fs.realizationList(RealizationStateEnum.STATE_HAS_DATA)
    assert 0 in realizations_list
    assert 8 in realizations_list
    assert 9 in realizations_list

    realizations_list = fs.realizationList(RealizationStateEnum.STATE_LOAD_FAILURE)
    assert 1 in realizations_list
    assert 2 in realizations_list
    assert 3 in realizations_list
    assert 4 in realizations_list
    assert 5 in realizations_list
    assert 6 in realizations_list
    assert 7 in realizations_list
