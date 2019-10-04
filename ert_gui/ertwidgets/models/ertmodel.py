from res.enkf import RealizationStateEnum, EnkfVarType
from res.job_queue import WorkflowRunner
from ecl.util.util import BoolVector, StringList
from ert_shared import ERT
from ert_gui.ertwidgets import showWaitCursorWhileWaiting


def getRealizationCount():
    return ERT.enkf_facade.get_ensemble_size()


def getAllCases():
    """ @rtype: list[str] """    
    return ERT.enkf_facade.get_all_cases()


def caseExists(case_name):
    """ @rtype: bool """
    return str(case_name) in getAllCases()


def caseIsInitialized(case_name):
    """ @rtype: bool """
    return ERT.enkf_facade.is_case_initialized(case_name)


def getAllInitializedCases():
    """ @rtype: list[str] """
    return [case for case in getAllCases() if caseIsInitialized(case)]


def getCurrentCaseName():
    """ @rtype: str """
    return ERT.enkf_facade.get_current_case_name()


def getHistoryLength():
    """ @rtype: int """
    return ERT.enkf_facade.get_history_length()


@showWaitCursorWhileWaiting
def selectOrCreateNewCase(case_name):
    if getCurrentCaseName() != case_name:
        ERT.enkf_facade.switch_file_system(case_name)
        ERT.emitErtChange()


def caseHasDataAndIsNotRunning(case):
    """ @rtype: bool """
    case_has_data = False
    state_map = ERT.enkf_facade.get_state_map(case)

    for state in state_map:
        if state == RealizationStateEnum.STATE_HAS_DATA:
            case_has_data = True
            break

    return case_has_data and not caseIsRunning(case)


def getAllCasesWithDataAndNotRunning():
    """ @rtype: list[str] """
    return [case for case in getAllCases() if caseHasDataAndIsNotRunning(case)]


def caseIsRunning(case):
    """ @rtype: bool """
    return ERT.enkf_facade.is_case_running(case)


def getAllCasesNotRunning():
    """ @rtype: list[str] """
    return [case for case in getAllCases() if not caseIsRunning(case)]


def getCaseRealizationStates(case_name):
    """ @rtype: list[res.enkf.enums.RealizationStateEnum] """
    state_map = ERT.enkf_facade.get_state_map(case_name)
    return [state for state in state_map]


@showWaitCursorWhileWaiting
def initializeCurrentCaseFromScratch(parameters, members):
    selected_parameters = StringList(parameters)
    mask = BoolVector(initial_size = getRealizationCount(), default_value = False)
    for member in members:
        member = int(member.strip())
        mask[member] = True

    ERT.enkf_facade.initialize_from_scratch(mask, selected_parameters)
    ERT.emitErtChange()


@showWaitCursorWhileWaiting
def initializeCurrentCaseFromExisting(source_case, target_case, source_report_step, parameters, members):
    if caseExists(source_case) and caseIsInitialized(source_case) and caseExists(target_case):
        total_member_count = getRealizationCount()

        member_mask = BoolVector.createFromList(total_member_count, members)
        selected_parameters = StringList(parameters)

        ERT.enkf_facade.initialize_from_existing_case(source_case, source_report_step, member_mask, selected_parameters)

        ERT.emitErtChange()


def getParameterList():
    """ @rtype: list[str] """
    return [str(p) for p in ERT.enkf_facade.get_keylist_from_var_type(EnkfVarType.PARAMETER)]


def getRunPath():
    """ @rtype: str """
    return ERT.enkf_facade.get_runpath_as_string()


def getNumberOfIterations():
    """ @rtype: int """
    return ERT.enkf_facade.get_number_of_iterations()


def setNumberOfIterations(iteration_count):
    """ @type iteration_count: int """
    if iteration_count != getNumberOfIterations():
        ERT.enkf_facade.set_number_of_iterations(iteration_count)
        ERT.emitErtChange()


def getWorkflowNames():
    """ @rtype: list[str] """
    return sorted(ERT.enkf_facade.get_workflow_list().getWorkflowNames(), key=str.lower)


def createWorkflowRunner(workflow_name):
    """ @rtype: WorkflowRunner """
    workflow_list = ERT.enkf_facade.get_workflow_list()

    workflow = workflow_list[workflow_name]
    context = workflow_list.getContext()
    return WorkflowRunner(workflow, ERT.ert, context)


def getAnalysisModules(iterable=False):
    """ @rtype: list[ert.analysis.AnalysisModule]"""
    return ERT.enkf_facade.get_analysis_modules(iterable)


def getAnalysisModuleNames(iterable=False):
    """ @rtype: list[str] """
    return ERT.enkf_facade.get_analysis_module_names(iterable)


def getCurrentAnalysisModuleName():
    """ @rtype: str """
    return ERT.enkf_facade.get_active_module_name()


def getQueueConfig():
    return ERT.enkf_facade.get_queue_config()
