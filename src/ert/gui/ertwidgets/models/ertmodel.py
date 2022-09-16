from typing import List

from ert._c_wrappers.enkf import EnKFMain, RealizationStateEnum, RunContext
from ert.gui.ertwidgets import showWaitCursorWhileWaiting


def get_runnable_realizations_mask(ert, casename):
    """Return the list of IDs corresponding to realizations that can be run.

    A realization is considered "runnable" if its status is any other than
    STATE_PARENT_FAILED. In that case, ERT does not know why that realization
    failed, so it does not even know whether the parameter set for that
    realization is sane or not.
    If the requested case does not exist, an empty list is returned
    """
    fsm = ert.getEnkfFsManager()
    if not fsm.caseExists(casename):
        return []
    sm = fsm.getStateMapForCase(casename)
    runnable_flag = (
        RealizationStateEnum.STATE_UNDEFINED
        | RealizationStateEnum.STATE_INITIALIZED
        | RealizationStateEnum.STATE_LOAD_FAILURE
        | RealizationStateEnum.STATE_HAS_DATA
    )
    return sm.createMask(runnable_flag)


@showWaitCursorWhileWaiting
def initializeCurrentCaseFromScratch(
    parameters: List[str], members: List[str], ert: EnKFMain
):
    mask = [False] * ert.getEnsembleSize()
    for member in members:
        mask[int(member.strip())] = True

    sim_fs = ert.getEnkfFsManager().getCurrentFileSystem()
    run_context = RunContext(sim_fs=sim_fs, mask=mask)
    ert.getEnkfFsManager().initRun(run_context, parameters=parameters)


@showWaitCursorWhileWaiting
def initializeCurrentCaseFromExisting(
    source_case: str,
    target_case: str,
    source_report_step: int,
    parameters: List[str],
    members: List[str],
    ert: EnKFMain,
):
    if (
        ert.caseExists(source_case)
        and ert.getEnkfFsManager().isCaseInitialized(source_case)
        and ert.caseExists(target_case)
    ):
        member_mask = [False] * ert.getEnsembleSize()
        for member in members:
            member_mask[int(member)] = True

        ert.getEnkfFsManager().customInitializeCurrentFromExistingCase(
            source_case, source_report_step, member_mask, parameters
        )
