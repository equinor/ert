from typing import List

from ert_shared.libres_facade import LibresFacade
from res.enkf import EnKFMain
from res.enkf import ErtRunContext
from ecl.util.util import BoolVector, StringList
from ert_gui.ertwidgets import showWaitCursorWhileWaiting


def getAllCases(facade: LibresFacade):
    """@rtype: list[str]"""
    case_list = facade.cases()
    return [str(case) for case in case_list if not facade.is_case_hidden(case)]


def caseExists(case_name, facade: LibresFacade):
    """@rtype: bool"""
    return str(case_name) in getAllCases(facade)


@showWaitCursorWhileWaiting
def initializeCurrentCaseFromScratch(
    parameters: List[str], members: List[str], ert: EnKFMain
):
    selected_parameters = StringList(parameters)
    mask = BoolVector(initial_size=ert.getEnsembleSize(), default_value=False)
    for member in members:
        member = int(member.strip())
        mask[member] = True

    sim_fs = ert.getEnkfFsManager().getCurrentFileSystem()
    run_context = ErtRunContext.case_init(sim_fs, mask)
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
        total_member_count = ert.getEnsembleSize()

        member_mask = BoolVector.createFromList(total_member_count, members)

        ert.getEnkfFsManager().customInitializeCurrentFromExistingCase(
            source_case, source_report_step, member_mask, parameters
        )
