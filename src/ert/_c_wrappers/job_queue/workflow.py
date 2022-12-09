import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from cwrap import BaseCClass  # pylint: disable=import-error

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.job_queue.workflow_joblist import WorkflowJoblist

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain
    from ert._c_wrappers.job_queue import WorkflowJob
    from ert._c_wrappers.util import SubstitutionList


class Workflow(BaseCClass):
    TYPE_NAME = "workflow"
    _alloc = ResPrototype("void* workflow_alloc(char*, workflow_joblist)", bind=False)
    _free = ResPrototype("void     workflow_free(workflow)")
    _count = ResPrototype("int      workflow_size(workflow)")
    _iget_job = ResPrototype("workflow_job_ref workflow_iget_job(workflow, int)")
    _iget_args = ResPrototype("stringlist_ref   workflow_iget_arguments(workflow, int)")

    _try_compile = ResPrototype("bool workflow_try_compile(workflow, subst_list)")
    _get_src_file = ResPrototype("char* worflow_get_src_file(workflow)")

    def __init__(self, src_file: str, job_list: WorkflowJoblist):
        c_ptr = self._alloc(src_file, job_list)
        super().__init__(c_ptr)

        self.__running = False
        self.__cancelled = False
        self.__current_job = None
        self.__status: Dict[str, Any] = {}

    def __len__(self):
        return self._count()

    def __getitem__(self, index: int) -> Tuple["WorkflowJob", Any]:
        job = self._iget_job(index)
        args = self._iget_args(index)
        return job, args

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @property
    def src_file(self):
        return self._get_src_file()

    def run(
        self,
        ert: "EnKFMain",
        verbose: bool = False,
        context: Optional["SubstitutionList"] = None,
    ) -> bool:
        logger = logging.getLogger(__name__)

        # Reset status
        self.__status = {}
        self.__running = True
        success = self._try_compile(context)
        if not success:
            msg = (
                "** Warning: The workflow file {} is not valid - "
                "make sure the workflow jobs are defined accordingly\n"
            )
            sys.stderr.write(msg.format(self.src_file))

            self.__running = False
            return False

        for job, args in self:
            self.__current_job = job
            if not self.__cancelled:
                return_value = job.run(ert, args, verbose)
                self.__status[job.name()] = {
                    "stdout": job.stdoutdata(),
                    "stderr": job.stderrdata(),
                    "completed": not job.hasFailed(),
                    "return": return_value,
                }

                info = {
                    "class": "WORKFLOW_JOB",
                    "job_name": job.name(),
                    "arguments": " ".join(args),
                    "stdout": job.stdoutdata(),
                    "stderr": job.stderrdata(),
                    "execution_type": job.execution_type,
                }

                if job.hasFailed():
                    logger.error(f"Workflow job {job.name()} failed", extra=info)
                else:
                    logger.info(
                        f"Workflow job {job.name()} completed successfully", extra=info
                    )

        self.__current_job = None
        self.__running = False
        return success

    def free(self):
        self._free()

    def isRunning(self):
        return self.__running

    def cancel(self):
        if self.__current_job is not None:
            self.__current_job.cancel()

        self.__cancelled = True

    def isCancelled(self):
        return self.__cancelled

    def wait(self):
        while self.isRunning():
            time.sleep(1)

    def getLastError(self) -> List[str]:
        return _clib.workflow.get_last_error(self)

    def getJobsReport(self) -> Dict[str, Any]:
        return self.__status

    @classmethod
    def createCReference(cls, c_pointer, parent=None):
        workflow = super().createCReference(c_pointer, parent)
        workflow.__running = False
        workflow.__cancelled = False
        workflow.__current_job = None
        return workflow

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        return os.path.realpath(self.src_file) == os.path.realpath(other.src_file)
