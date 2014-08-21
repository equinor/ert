from ert.cwrap import BaseCClass, CWrapper
from ert.job_queue import JOB_QUEUE_LIB
from ert.config import ContentTypeEnum


class WorkflowJob(BaseCClass):

    def __init__(self, name, internal=True):
        c_ptr = WorkflowJob.cNamespace().alloc(name, internal)
        super(WorkflowJob, self).__init__(c_ptr)

    def isInternal(self):
        """ @rtype: bool """
        return WorkflowJob.cNamespace().internal(self)

    def name(self):
        """ @rtype: str """
        return WorkflowJob.cNamespace().name(self)

    def minimumArgumentCount(self):
        """ @rtype: int """
        return WorkflowJob.cNamespace().min_arg(self)

    def maximumArgumentCount(self):
        """ @rtype: int """
        return WorkflowJob.cNamespace().max_arg(self)


    def isInternalScript(self):
        """ @rtype: bool """
        return WorkflowJob.cNamespace().is_internal_script(self)

    def getInternalScriptPath(self):
        """ @rtype: str """
        return WorkflowJob.cNamespace().get_internal_script(self)

    def argumentTypes(self):
        """ @rtype: list of ContentTypeEnum """

        result = []
        for index in range(self.maximumArgumentCount()):
            t = WorkflowJob.cNamespace().arg_type(self, index)
            if t == ContentTypeEnum.CONFIG_BOOL:
                result.append(bool)
            elif t == ContentTypeEnum.CONFIG_FLOAT:
                result.append(float)
            elif t == ContentTypeEnum.CONFIG_INT:
                result.append(int)
            elif t == ContentTypeEnum.CONFIG_STRING:
                result.append(str)
            else:
                result.append(None)

        return result


    def run(self, monitor, ert, verbose, arguments):
        """
        @type monitor: ert.job_queue.workflow_job_monitor.WorkflowJobMonitor
        @type ert: ert.enkf.enkf_main.EnKFMain
        @type verbose: bool
        @type arguments: StringList
        @rtype: ctypes.c_void_p
        """

        if self.isInternalScript():
            print("Running jobs of this type is not yet supported!")
            return None
        else:
            return WorkflowJob.cNamespace().run(self, monitor, ert, verbose, arguments)

    def free(self):
        WorkflowJob.cNamespace().free(self)


CWrapper.registerObjectType("workflow_job", WorkflowJob)

cwrapper = CWrapper(JOB_QUEUE_LIB)

WorkflowJob.cNamespace().alloc    = cwrapper.prototype("c_void_p workflow_job_alloc(char*, bool)")
WorkflowJob.cNamespace().free     = cwrapper.prototype("void     workflow_job_free(workflow_job)")
WorkflowJob.cNamespace().name     = cwrapper.prototype("char*    workflow_job_get_name(workflow_job)")
WorkflowJob.cNamespace().internal = cwrapper.prototype("bool     workflow_job_internal(workflow_job)")
WorkflowJob.cNamespace().is_internal_script  = cwrapper.prototype("bool   workflow_job_is_internal_script(workflow_job)")
WorkflowJob.cNamespace().get_internal_script = cwrapper.prototype("char*  workflow_job_get_internal_script_path(workflow_job)")

WorkflowJob.cNamespace().min_arg  = cwrapper.prototype("int  workflow_job_get_min_arg(workflow_job)")
WorkflowJob.cNamespace().max_arg  = cwrapper.prototype("int  workflow_job_get_max_arg(workflow_job)")
WorkflowJob.cNamespace().arg_type = cwrapper.prototype("config_content_type_enum workflow_job_iget_argtype(workflow_job, int)")

WorkflowJob.cNamespace().run = cwrapper.prototype("c_void_p workflow_job_run(workflow_job, workflow_job_monitor, c_void_p, bool, stringlist)")