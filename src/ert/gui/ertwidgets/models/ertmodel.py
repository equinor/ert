from ert._c_wrappers.enkf import RealizationStateEnum


def get_runnable_realizations_mask(ert, casename):
    """Return the list of IDs corresponding to realizations that can be run.

    A realization is considered "runnable" if its status is any other than
    STATE_PARENT_FAILED. In that case, ERT does not know why that realization
    failed, so it does not even know whether the parameter set for that
    realization is sane or not.
    If the requested case does not exist, an empty list is returned
    """
    fsm = ert.storage_manager
    if casename not in fsm:
        return []
    sm = fsm.state_map(casename)
    runnable_flag = (
        RealizationStateEnum.STATE_UNDEFINED
        | RealizationStateEnum.STATE_INITIALIZED
        | RealizationStateEnum.STATE_LOAD_FAILURE
        | RealizationStateEnum.STATE_HAS_DATA
    )
    return sm.createMask(runnable_flag)
