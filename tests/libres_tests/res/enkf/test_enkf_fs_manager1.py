from res.enkf import EnKFMain


def test_enkf_fs_manager_create(setup_case):
    # We are indirectly testing the create through the create
    # already in the enkf_main object. In principle we could
    # create a separate manager instance from the ground up, but
    # then the reference count will be weird.
    res_config = setup_case("local/snake_oil", "snake_oil.ert")
    ert = EnKFMain(res_config)
    fsm = ert.getEnkfFsManager()

    fsm.getCurrentFileSystem()
    assert fsm.isCaseMounted("default_0")
    assert fsm.caseExists("default_0")
    assert fsm.caseHasData("default_0")
    assert not fsm.isCaseRunning("default_0")

    assert fsm.getFileSystemCount() == 1

    assert not fsm.isCaseMounted("newFS")
    assert not fsm.caseExists("newFS")
    assert not fsm.caseHasData("newFS")
    assert not fsm.isCaseRunning("newFS")

    fsm.getFileSystem("newFS")
    assert fsm.getFileSystemCount() == 2

    assert fsm.isCaseMounted("newFS")
    assert fsm.caseExists("newFS")
    assert not fsm.caseHasData("newFS")
    assert not fsm.isCaseRunning("newFS")
