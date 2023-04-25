from __future__ import annotations

import logging
from concurrent import futures
from typing import TYPE_CHECKING, Any, Dict, Optional

from ert._c_wrappers.job_queue import Workflow

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain
    from ert.storage import EnsembleAccessor, StorageAccessor


class WorkflowRunner:
    def __init__(
        self,
        workflow: Workflow,
        ert: Optional[EnKFMain] = None,
        storage: Optional[StorageAccessor] = None,
        ensemble: Optional[EnsembleAccessor] = None,
    ):
        self.__workflow = workflow
        self._ert = ert
        self._storage = storage
        self._ensemble = ensemble

        self.__workflow_result = None
        self._workflow_executor = futures.ThreadPoolExecutor(max_workers=1)
        self._workflow_job = None

        self.__running = False
        self.__cancelled = False
        self.__current_job = None
        self.__status: Dict[str, Any] = {}

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, _type, value, traceback):
        self.wait()

    def run(self):
        if self.isRunning():
            raise AssertionError("An instance of workflow is already running!")
        self._workflow_job = self._workflow_executor.submit(self.run_blocking)

    def run_blocking(self) -> None:
        self.__workflow_result = None
        logger = logging.getLogger(__name__)

        # Reset status
        self.__status = {}
        self.__running = True

        for job, args in self.__workflow:
            self.__current_job = job
            if not self.__cancelled:
                logger.info(f"Workflow job {job.name} starting")
                job.run(self._ert, self._storage, self._ensemble, args)
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
        self.__workflow_result = True

    def isRunning(self) -> bool:
        if self.__running:
            return True

        # Completion of _workflow does not indicate that __workflow_result is
        # set. Check future status, since __workflow_result follows future
        # completion.
        return self._workflow_job is not None and not self._workflow_job.done()

    def isCancelled(self) -> bool:
        return self.__cancelled

    def cancel(self):
        if self.isRunning():
            if self.__current_job is not None:
                self.__current_job.cancel()

            self.__cancelled = True
        self.wait()

    def exception(self):
        if self._workflow_job is not None:
            return self._workflow_job._exception
        return None

    def wait(self):
        # This returns a tuple (done, pending), since we run only one job we don't
        # need to use it
        _, _ = futures.wait(
            [self._workflow_job], timeout=None, return_when=futures.FIRST_EXCEPTION
        )

    def workflowResult(self) -> Optional[bool]:
        return self.__workflow_result

    def workflowReport(self):
        """@rtype: {dict}"""
        return self.__status
