import logging
import os
import sys
import time
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ert._c_wrappers.job_queue.workflow_joblist import WorkflowJoblist

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain
    from ert._c_wrappers.job_queue import WorkflowJob
    from ert._c_wrappers.util import SubstitutionList


class Workflow:
    def __init__(self, src_file: str, job_list: WorkflowJoblist):
        self.__running = False
        self.__cancelled = False
        self.__current_job = None
        self.__status: Dict[str, Any] = {}
        self._compile_time = None
        self.job_list = job_list
        self.src_file = src_file
        self.cmd_list = []
        self.last_error = []

        self._try_compile(None)

    def __len__(self):
        return len(self.cmd_list)

    @property
    def compiled(self):
        return self._compile_time is None

    def __getitem__(self, index: int) -> Tuple["WorkflowJob", Any]:
        return self.cmd_list[index]

    def __iter__(self):
        return iter(self.cmd_list)

    def _try_compile(self, context: Optional["SubstitutionList"]):
        to_compile = self.src_file
        if not os.path.exists(self.src_file):
            raise ValueError(f"Workflow file {self.src_file} does not exist")
        if context is not None:
            tmpdir = mkdtemp("ert_workflow")
            to_compile = os.path.join(tmpdir, "ert-workflow")
            context.substitute_file(self.src_file, to_compile)
            self._compile_time = None
        mtime = os.path.getmtime(to_compile)
        if self._compile_time is not None and mtime <= self._compile_time:
            return

        self.cmd_list = []
        parser = self.job_list.parser
        content = parser.parse(to_compile)

        errors = content.getErrors()
        if errors:
            return errors

        for line in content:
            self.cmd_list.append(
                (
                    self.job_list[line.get_kw()],
                    [line.igetString(i) for i in range(len(line))],
                )
            )

        self._compile_time = mtime
        return None

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
        errors = self._try_compile(context)
        if errors is not None:
            msg = (
                "** Warning: The workflow file {} is not valid - "
                "make sure the workflow jobs are defined correctly:\n {}\n"
            )
            sys.stderr.write(msg.format(self.src_file, errors))

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
        return True

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
        return self.last_error

    def getJobsReport(self) -> Dict[str, Any]:
        return self.__status

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        return os.path.realpath(self.src_file) == os.path.realpath(other.src_file)
