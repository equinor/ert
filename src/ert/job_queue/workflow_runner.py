from __future__ import annotations

import logging
import os
from concurrent import futures
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from typing_extensions import Self

from ert.config import ErtScript, ExternalErtScript, Workflow, WorkflowJob

if TYPE_CHECKING:
    from ert.enkf_main import EnKFMain
    from ert.storage import EnsembleAccessor, StorageAccessor


class WorkflowJobRunner:
    def __init__(self, workflow_job: WorkflowJob):
        self.job = workflow_job
        self.__running = False
        self.__script: Optional[ErtScript] = None
        self.stop_on_fail = False

    def run(
        self,
        ert: Optional[EnKFMain] = None,
        storage: Optional[StorageAccessor] = None,
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
            if self.job.stop_on_fail is not None:
                self.stop_on_fail = self.job.stop_on_fail
            elif self.__script is not None:
                self.stop_on_fail = self.__script.stop_on_fail or False

        elif not self.job.internal:
            self.__script = ExternalErtScript(
                ert,  # type: ignore
                storage,  # type: ignore
                self.job.executable,  # type: ignore
            )

            if self.job.stop_on_fail is not None:
                self.stop_on_fail = self.job.stop_on_fail
            elif self.job.executable is not None and os.path.isfile(
                self.job.executable
            ):
                try:
                    with open(self.job.executable, encoding="utf-8") as executable:
                        lines = executable.readlines()
                        if any(
                            line.lower().replace(" ", "").replace("\n", "")
                            == "stop_on_fail=true"
                            for line in lines
                        ):
                            self.stop_on_fail = True
                except Exception:  # pylint: disable=broad-exception-caught
                    self.stop_on_fail = False

        else:
            raise UserWarning("Unknown script type!")
        result = self.__script.initializeAndRun(  # type: ignore
            self.job.argument_types(),
            arguments,
        )
        self.__running = False

        return result

    @property
    def name(self) -> str:
        return self.job.name

    @property
    def execution_type(self) -> str:
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
    ) -> None:
        self.__workflow = workflow
        self._ert = ert
        self._storage = storage
        self._ensemble = ensemble

        self.__workflow_result: Optional[bool] = None
        self._workflow_executor = futures.ThreadPoolExecutor(max_workers=1)
        self._workflow_job: Optional[Future[None]] = None

        self.__running = False
        self.__cancelled = False
        self.__current_job: Optional[WorkflowJobRunner] = None
        self.__status: Dict[str, Dict[str, Any]] = {}

    def __enter__(self) -> Self:
        self.run()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        self.wait()

    def run(self) -> None:
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
            jobrunner = WorkflowJobRunner(job)
            self.__current_job = jobrunner
            if not self.__cancelled:
                logger.info(f"Workflow job {jobrunner.name} starting")
                jobrunner.run(self._ert, self._storage, self._ensemble, args)
                self.__status[jobrunner.name] = {
                    "stdout": jobrunner.stdoutdata(),
                    "stderr": jobrunner.stderrdata(),
                    "completed": not jobrunner.hasFailed(),
                }

                info = {
                    "class": "WORKFLOW_JOB",
                    "job_name": jobrunner.name,
                    "arguments": " ".join(args),
                    "stdout": jobrunner.stdoutdata(),
                    "stderr": jobrunner.stderrdata(),
                    "execution_type": jobrunner.execution_type,
                }

                if jobrunner.hasFailed():
                    if jobrunner.stop_on_fail:
                        self.__running = False
                        raise RuntimeError(
                            f"Workflow job {info['job_name']}"
                            f" failed with error: {info['stderr']}"
                        )

                    logger.error(f"Workflow job {jobrunner.name} failed", extra=info)
                else:
                    logger.info(
                        f"Workflow job {jobrunner.name} completed successfully",
                        extra=info,
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

    def cancel(self) -> None:
        if self.isRunning():
            if self.__current_job is not None:
                self.__current_job.cancel()

            self.__cancelled = True
        self.wait()

    def exception(self) -> Optional[BaseException]:
        if self._workflow_job is not None:
            return self._workflow_job.exception()
        return None

    def wait(self) -> None:
        # This returns a tuple (done, pending), since we run only one job we don't
        # need to use it
        if self._workflow_job is not None:
            _, _ = futures.wait(
                [self._workflow_job], timeout=None, return_when=futures.FIRST_EXCEPTION
            )

    def workflowResult(self) -> Optional[bool]:
        return self.__workflow_result

    def workflowReport(self) -> Dict[str, Dict[str, Any]]:
        return self.__status
