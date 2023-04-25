from __future__ import annotations

import logging
from concurrent import futures
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ert._c_wrappers.job_queue import ExternalErtScript, Workflow, WorkflowJob

from .ert_script import ErtScript

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain
    from ert.storage import EnsembleAccessor, StorageAccessor


class WorkflowJobRunner:
    def __init__(self, workflow_job: WorkflowJob):
        self.job = workflow_job
        self.__running = False
        self.__script: Optional[ErtScript] = None

    def run(
        self,
        ert: EnKFMain,
        storage: StorageAccessor,
        ensemble: Optional[EnsembleAccessor] = None,
        arguments: Optional[List[Any]] = None,
    ) -> Any:
        if arguments is None:
            arguments = []
        self.__running = True
        if self.job.min_args and len(arguments) < self.job.min_args:
            raise ValueError(
                f"The job: {self.job.name} requires at least "
                f"{self.job.min_args} arguments, {len(arguments)} given."
            )

        if self.job.max_args and self.job.max_args < len(arguments):
            raise ValueError(
                f"The job: {self.job.name} can only have "
                f"{self.job.max_args} arguments, {len(arguments)} given."
            )

        if self.job.ert_script is not None:
            self.__script = self.job.ert_script(ert, storage, ensemble)
        elif not self.job.internal:
            self.__script = ExternalErtScript(ert, storage, self.job.executable)
        else:
            raise UserWarning("Unknown script type!")
        result = self.__script.initializeAndRun(self.job.argumentTypes(), arguments)
        self.__running = False
        return result

    @property
    def name(self):
        return self.job.name

    @property
    def execution_type(self):
        if self.job.internal and self.job.script is not None:
            return "internal python"
        elif self.job.internal:
            return "internal C"
        return "external"

    def cancel(self) -> None:
        if self.__script is not None:
            self.__script.cancel()

    def isRunning(self) -> bool:
        return self.__running

    def isCancelled(self) -> bool:
        if self.__script is None:
            raise ValueError("The job must be run before calling isCancelled")
        return self.__script.isCancelled()

    def hasFailed(self) -> bool:
        if self.__script is None:
            raise ValueError("The job must be run before calling hasFailed")
        return self.__script.hasFailed()

    def stdoutdata(self) -> str:
        if self.__script is None:
            raise ValueError("The job must be run before getting stdoutdata")
        return self.__script.stdoutdata

    def stderrdata(self) -> str:
        if self.__script is None:
            raise ValueError("The job must be run before getting stderrdata")
        return self.__script.stderrdata


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
            job = WorkflowJobRunner(job)
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
