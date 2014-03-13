from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB



class ErtWorkflowListHandler(BaseCClass):
    def __init__(self, workflow_list, workflow_name, enkf_main_pointer):
        pointer = ErtWorkflowListHandler.cNamespace().alloc()
        super(ErtWorkflowListHandler, self).__init__(pointer)

        self.__workflow_list = workflow_list
        self.__workflow_name = workflow_name
        self.__enkf_main_pointer = enkf_main_pointer

    def isRunning(self):
        return ErtWorkflowListHandler.cNamespace().is_running(self)

    def getWorkflowResult(self):
        return ErtWorkflowListHandler.cNamespace().read_result(self)

    def runWorkflow(self):
        ErtWorkflowListHandler.cNamespace().run_workflow(self, self.__workflow_list, self.__workflow_name, self.__enkf_main_pointer)

    def free(self):
        ErtWorkflowListHandler.cNamespace().free(self)

    def cancelWorkflow(self):
        ErtWorkflowListHandler.cNamespace().cancel_workflow(self)
        ErtWorkflowListHandler.cNamespace().join_workflow(self)

    def isKilled(self):
        return ErtWorkflowListHandler.cNamespace().is_killed(self)

cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerType("ert_workflow_thread_data", ErtWorkflowListHandler)
cwrapper.registerType("ert_workflow_thread_data_ref", ErtWorkflowListHandler.createCReference)
cwrapper.registerType("ert_workflow_thread_data_obj", ErtWorkflowListHandler.createPythonObject)

ErtWorkflowListHandler.cNamespace().alloc = cwrapper.prototype("c_void_p ert_workflow_list_handler_alloc()")
ErtWorkflowListHandler.cNamespace().free = cwrapper.prototype("void ert_workflow_list_handler_free(ert_workflow_thread_data)")

ErtWorkflowListHandler.cNamespace().run_workflow = cwrapper.prototype("void ert_workflow_list_handler_run_workflow(ert_workflow_thread_data, ert_workflow_list, char*, c_void_p)")
ErtWorkflowListHandler.cNamespace().read_result = cwrapper.prototype("bool ert_workflow_list_handler_read_result(ert_workflow_thread_data)")
ErtWorkflowListHandler.cNamespace().is_running = cwrapper.prototype("bool ert_workflow_list_handler_is_running(ert_workflow_thread_data)")
ErtWorkflowListHandler.cNamespace().is_killed = cwrapper.prototype("bool ert_workflow_list_handler_is_killed(ert_workflow_thread_data)")
ErtWorkflowListHandler.cNamespace().cancel_workflow = cwrapper.prototype("void ert_workflow_list_handler_stop_workflow(ert_workflow_thread_data)")
ErtWorkflowListHandler.cNamespace().join_workflow = cwrapper.prototype("void ert_workflow_list_handler_join_workflow(ert_workflow_thread_data)")
