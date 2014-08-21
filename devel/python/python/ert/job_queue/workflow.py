from ert.cwrap import BaseCClass, CWrapper
from ert.job_queue import JOB_QUEUE_LIB, WorkflowJoblist, WorkflowJob, WorkflowJobMonitor
from ert.util import SubstitutionList

class Workflow(BaseCClass):

    def __init__(self, src_file, job_list):
        """
        @type src_file: str
        @type job_list: WorkflowJoblist
        """
        c_ptr = Workflow.cNamespace().alloc(src_file, job_list)
        super(Workflow, self).__init__(c_ptr)

    def __len__(self):
        return Workflow.cNamespace().count(self)

    def __getitem__(self, index):
        """
        @type index: int
        @rtype: tuple of (WorkflowJob, arguments)
        """
        job = Workflow.cNamespace().iget_job(self, index)
        args = Workflow.cNamespace().iget_args(self, index)
        return job, args

    def __iter__(self):
        index = 0

        for index in range(len(self)):
            yield self[index]
            index += 1

    def run(self, monitor, ert, verbose=False, context=None):
        """
        @type monitor: WorkflowJobMonitor
        @type ert: ert.enkf.enkf_main.EnKFMain
        @type verbose: bool
        @type context: SubstitutionList
        @rtype: bool
        """
        success = Workflow.cNamespace().try_compile(self, context)

        if success:
            for job, args in self:
                return_value = job.run(monitor, ert, verbose, args)

        return success


    def free(self):
        Workflow.cNamespace().free(self)


CWrapper.registerObjectType("workflow", Workflow)

cwrapper = CWrapper(JOB_QUEUE_LIB)

Workflow.cNamespace().alloc = cwrapper.prototype("c_void_p workflow_alloc(char*, workflow_joblist)")
Workflow.cNamespace().free  = cwrapper.prototype("void     workflow_free(workflow)")
Workflow.cNamespace().count = cwrapper.prototype("int      workflow_size(workflow)")
Workflow.cNamespace().iget_job   = cwrapper.prototype("workflow_job_ref workflow_iget_job(workflow, int)")
Workflow.cNamespace().iget_args  = cwrapper.prototype("stringlist_ref   workflow_iget_arguments(workflow, int)")

Workflow.cNamespace().try_compile = cwrapper.prototype("bool workflow_try_compile(workflow, subst_list)")
