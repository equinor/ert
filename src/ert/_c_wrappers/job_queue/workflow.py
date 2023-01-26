import logging
import os
import time
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ert._c_wrappers.config import ConfigParser, ConfigValidationError

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain
    from ert._c_wrappers.job_queue import WorkflowJob
    from ert._c_wrappers.util import SubstitutionList


def _workflow_parser(workflow_jobs: Dict[str, "WorkflowJob"]) -> ConfigParser:
    parser = ConfigParser()
    for name, job in workflow_jobs.items():
        item = parser.add(name)
        item.set_argc_minmax(job.min_args, job.max_args)
        for i, t in enumerate(job.arg_types):
            item.iset_type(i, t)
    return parser


class Workflow:
    def __init__(
        self,
        src_file: str,
        cmd_list: List["WorkflowJob"],
    ):
        self.__running = False
        self.__cancelled = False
        self.__current_job = None
        self.__status: Dict[str, Any] = {}
        self.src_file = src_file
        self.cmd_list = cmd_list

    def __len__(self):
        return len(self.cmd_list)

    def __getitem__(self, index: int) -> Tuple["WorkflowJob", Any]:
        return self.cmd_list[index]

    def __iter__(self):
        return iter(self.cmd_list)

    @classmethod
    def from_file(
        cls,
        src_file: str,
        context: Optional["SubstitutionList"],
        job_list: Dict[str, "WorkflowJob"],
    ):
        to_compile = src_file
        if not os.path.exists(src_file):
            raise ConfigValidationError(f"Workflow file {src_file} does not exist")
        if context is not None:
            tmpdir = mkdtemp("ert_workflow")
            to_compile = os.path.join(tmpdir, "ert-workflow")
            context.substitute_file(src_file, to_compile)

        cmd_list = []
        parser = _workflow_parser(job_list)
        content = parser.parse(to_compile)

        for line in content:
            cmd_list.append(
                (
                    job_list[line.get_kw()],
                    [line.igetString(i) for i in range(len(line))],
                )
            )

        return cls(src_file, cmd_list)

    def run(self, ert: "EnKFMain") -> bool:
        logger = logging.getLogger(__name__)

        # Reset status
        self.__status = {}
        self.__running = True

        for job, args in self:
            self.__current_job = job
            if not self.__cancelled:
                job.run(ert, args)
                self.__status[job.name] = {
                    "stdout": job.stdoutdata(),
                    "stderr": job.stderrdata(),
                    "completed": not job.hasFailed(),
                }

                info = {
                    "class": "WORKFLOW_JOB",
                    "job_name": job.name,
                    "arguments": " ".join(args),
                    "stdout": job.stdoutdata(),
                    "stderr": job.stderrdata(),
                    "execution_type": job.execution_type,
                }

                if job.hasFailed():
                    logger.error(f"Workflow job {job.name} failed", extra=info)
                else:
                    logger.info(
                        f"Workflow job {job.name} completed successfully", extra=info
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

    def getJobsReport(self) -> Dict[str, Any]:
        return self.__status

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        return os.path.realpath(self.src_file) == os.path.realpath(other.src_file)
