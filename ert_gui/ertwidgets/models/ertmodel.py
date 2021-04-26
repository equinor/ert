from res.analysis.analysis_module import AnalysisModule
from res.analysis.enums.analysis_module_options_enum import AnalysisModuleOptionsEnum
from res.enkf import RealizationStateEnum, EnkfVarType
from res.enkf import ErtRunContext
from res.job_queue import WorkflowRunner
from ecl.util.util import BoolVector, StringList
from ert_shared import ERT
from ert_gui.ertwidgets import showWaitCursorWhileWaiting


def getRealizationCount():
    return ERT.enkf_facade.get_ensemble_size()


def getAllCases():
    """@rtype: list[str]"""
    case_list = ERT.ert.getEnkfFsManager().getCaseList()
    return [
        str(case)
        for case in case_list
        if not ERT.ert.getEnkfFsManager().isCaseHidden(case)
    ]


def caseExists(case_name):
    """@rtype: bool"""
    return str(case_name) in getAllCases()


def caseIsInitialized(case_name):
    """@rtype: bool"""
    return ERT.ert.getEnkfFsManager().isCaseInitialized(case_name)


def getAllInitializedCases():
    """@rtype: list[str]"""
    return [case for case in getAllCases() if caseIsInitialized(case)]


def getCurrentCaseName():
    """@rtype: str"""
    return ERT.enkf_facade.get_current_case_name()


def getHistoryLength():
    """@rtype: int"""
    return ERT.ert.getHistoryLength()


def get_runnable_realizations_mask(casename):
    """Return the list of IDs corresponding to realizations that can be run.

    A realization is considered "runnable" if its status is any other than
    STATE_PARENT_FAILED. In that case, ERT does not know why that realization
    failed, so it does not even know whether the parameter set for that
    realization is sane or not.
    If the requested case does not exist, an empty list is returned
    """
    fsm = ERT.ert.getEnkfFsManager()
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
def selectOrCreateNewCase(case_name):
    ERT.enkf_facade.select_or_create_new_case(case_name)
    ERT.emitErtChange()


def caseHasDataAndIsNotRunning(case):
    """@rtype: bool"""
    case_has_data = False
    state_map = ERT.ert.getEnkfFsManager().getStateMapForCase(case)

    for state in state_map:
        if state == RealizationStateEnum.STATE_HAS_DATA:
            case_has_data = True
            break

    return case_has_data and not caseIsRunning(case)


def getAllCasesWithDataAndNotRunning():
    """@rtype: list[str]"""
    return [case for case in getAllCases() if caseHasDataAndIsNotRunning(case)]


def caseIsRunning(case):
    """@rtype: bool"""
    return ERT.ert.getEnkfFsManager().isCaseRunning(case)


def getAllCasesNotRunning():
    """@rtype: list[str]"""
    return [case for case in getAllCases() if not caseIsRunning(case)]


def getCaseRealizationStates(case_name):
    """@rtype: list[res.enkf.enums.RealizationStateEnum]"""
    state_map = ERT.ert.getEnkfFsManager().getStateMapForCase(case_name)
    return [state for state in state_map]


@showWaitCursorWhileWaiting
def initializeCurrentCaseFromScratch(parameters, members):
    selected_parameters = StringList(parameters)
    mask = BoolVector(initial_size=getRealizationCount(), default_value=False)
    for member in members:
        member = int(member.strip())
        mask[member] = True

    sim_fs = ERT.ert.getEnkfFsManager().getCurrentFileSystem()
    run_context = ErtRunContext.case_init(sim_fs, mask)
    ERT.ert.getEnkfFsManager().initializeFromScratch(selected_parameters, run_context)
    ERT.emitErtChange()


@showWaitCursorWhileWaiting
def initializeCurrentCaseFromExisting(
    source_case, target_case, source_report_step, parameters, members
):
    if (
        caseExists(source_case)
        and caseIsInitialized(source_case)
        and caseExists(target_case)
    ):
        total_member_count = getRealizationCount()

        member_mask = BoolVector.createFromList(total_member_count, members)
        selected_parameters = StringList(parameters)

        ERT.ert.getEnkfFsManager().customInitializeCurrentFromExistingCase(
            source_case, source_report_step, member_mask, selected_parameters
        )

        ERT.emitErtChange()


def getParameterList():
    """@rtype: list[str]"""
    return [
        str(p)
        for p in ERT.ert.ensembleConfig().getKeylistFromVarType(EnkfVarType.PARAMETER)
    ]


def getRunPath():
    """@rtype: str"""
    return ERT.ert.getModelConfig().getRunpathAsString()


def getNumberOfIterations():
    """@rtype: int"""
    return ERT.enkf_facade.get_number_of_iterations()


def setNumberOfIterations(iteration_count):
    """@type iteration_count: int"""
    if iteration_count != getNumberOfIterations():
        ERT.ert.analysisConfig().getAnalysisIterConfig().setNumIterations(
            iteration_count
        )
        ERT.emitErtChange()


def getWorkflowNames():
    """@rtype: list[str]"""
    return sorted(ERT.ert.getWorkflowList().getWorkflowNames(), key=str.lower)


def createWorkflowRunner(workflow_name):
    """@rtype: WorkflowRunner"""
    workflow_list = ERT.ert.getWorkflowList()

    workflow = workflow_list[workflow_name]
    context = workflow_list.getContext()
    return WorkflowRunner(workflow, ERT.ert, context)


def getAnalysisModules(iterable=False):
    """@rtype: list[ert.analysis.AnalysisModule]"""
    return ERT.enkf_facade.get_analysis_modules(iterable)


def getAnalysisModuleNames(iterable=False):
    """@rtype: list[str]"""
    return ERT.enkf_facade.get_analysis_module_names(iterable)


def getCurrentAnalysisModuleName():
    """@rtype: str"""
    return ERT.ert.analysisConfig().activeModuleName()


def getQueueConfig():
    return ERT.enkf_facade.get_queue_config()
