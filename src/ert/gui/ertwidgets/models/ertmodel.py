from typing import List

from ert_shared.libres_facade import LibresFacade
from res.enkf import EnKFMain, RealizationStateEnum
from res.enkf import RunContext
from ecl.util.util import StringList
from ert.gui.ertwidgets import showWaitCursorWhileWaiting


def getAllCases(facade: LibresFacade):
    """@rtype: list[str]"""
    case_list = facade.cases()
    return [str(case) for case in case_list if not facade.is_case_hidden(case)]


def caseExists(case_name, facade: LibresFacade):
    """@rtype: bool"""
    return str(case_name) in getAllCases(facade)


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
    selected_parameters = StringList(parameters)
    mask = [False] * ert.getEnsembleSize()
    for member in members:
        member = int(member.strip())
        mask[member] = True

    sim_fs = ert.getEnkfFsManager().getCurrentFileSystem()
    run_context = RunContext(sim_fs=sim_fs, mask=mask)
    ert.getEnkfFsManager().initializeFromScratch(selected_parameters, run_context)


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
        caseExists(source_case, LibresFacade(ert))
        and ert.getEnkfFsManager().isCaseInitialized(source_case)
        and caseExists(target_case, LibresFacade(ert))
    ):
        member_mask = [False] * ert.getEnsembleSize()
        for member in members:
            member_mask[int(member)] = True

        ert.getEnkfFsManager().customInitializeCurrentFromExistingCase(
            source_case, source_report_step, member_mask, parameters
        )
