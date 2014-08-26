from ert.cwrap import BaseCClass, CWrapper
from ert.job_queue import JOB_QUEUE_LIB


class WorkflowJobMonitor(BaseCClass):

    def __init__(self):
        c_ptr = WorkflowJobMonitor.cNamespace().alloc()
        super(WorkflowJobMonitor, self).__init__(c_ptr)

    def setPID(self, pid):
        """
        @type pid: int
        """
        WorkflowJobMonitor.cNamespace().set_pid(self, pid)

    def getPID(self):
        """ @rtype: int """
        return WorkflowJobMonitor.cNamespace().get_pid(self)

    def setBlocking(self, blocking):
        """
        @type blocking: bool
        """
        WorkflowJobMonitor.cNamespace().set_blocking(self, blocking)

    def isBlocking(self):
        """ @rtype: bool """
        return WorkflowJobMonitor.cNamespace().is_blocking(self)

    def free(self):
        WorkflowJobMonitor.cNamespace().free(self)


CWrapper.registerObjectType("workflow_job_monitor", WorkflowJobMonitor)

cwrapper = CWrapper(JOB_QUEUE_LIB)

WorkflowJobMonitor.cNamespace().alloc    = cwrapper.prototype("c_void_p workflow_job_monitor_alloc()")
WorkflowJobMonitor.cNamespace().free     = cwrapper.prototype("void     workflow_job_monitor_free(workflow_job_monitor)")

WorkflowJobMonitor.cNamespace().set_pid = cwrapper.prototype("void workflow_job_monitor_set_pid(workflow_job_monitor, int)")
WorkflowJobMonitor.cNamespace().get_pid = cwrapper.prototype("int workflow_job_monitor_get_pid(workflow_job_monitor)")
WorkflowJobMonitor.cNamespace().set_blocking = cwrapper.prototype("void workflow_job_monitor_set_blocking(workflow_job_monitor, bool)")
WorkflowJobMonitor.cNamespace().is_blocking = cwrapper.prototype("bool workflow_job_monitor_get_blocking(workflow_job_monitor)")
