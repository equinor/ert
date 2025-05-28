from __future__ import annotations

import logging
from concurrent import futures
from concurrent.futures import Future
from typing import Any, Self

from ert.config import Workflow
from ert.config.workflow_job import ErtScriptWorkflow, _WorkflowJob
from ert.plugins import (
    ErtScript,
    ExternalErtScript,
    WorkflowFixtures,
)


class WorkflowJobRunner:
    def __init__(self, workflow_job: _WorkflowJob) -> None:
        self.job = workflow_job
        self.__running = False
        self.__script: ErtScript | None = None
        self.stop_on_fail = False

    def run(
        self,
        arguments: list[Any] | None = None,
        fixtures: WorkflowFixtures | None = None,
    ) -> Any:
        if arguments is None:
            arguments = []
        fixtures = {} if fixtures is None else fixtures
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

        if isinstance(self.job, ErtScriptWorkflow):
            self.__script = self.job.ert_script()
            # We let stop on fail either from class or config take precedence
            self.stop_on_fail = self.job.stop_on_fail or self.__script.stop_on_fail

        else:
            self.__script = ExternalErtScript(
                self.job.executable,  # type: ignore
            )
            self.stop_on_fail = self.job.stop_on_fail

        result = self.__script.initializeAndRun(
            self.job.argument_types(), arguments, fixtures
        )
        self.__running = False

        return result

    @property
    def name(self) -> str:
        return self.job.name

    @property
    def execution_type(self) -> str:
        if isinstance(self.job, ErtScriptWorkflow):
            return "internal python"
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
        fixtures: WorkflowFixtures,
    ) -> None:
        self.__workflow = workflow
        self.fixtures = fixtures

        self.__workflow_result: bool | None = None
        self._workflow_executor = futures.ThreadPoolExecutor(max_workers=1)
        self._workflow_job: Future[None] | None = None

        self.__running = False
        self.__cancelled = False
        self.__current_job: WorkflowJobRunner | None = None
        self.__status: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> Self:
        self.run()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
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
                jobrunner.run(args, fixtures=self.fixtures)
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

    def exception(self) -> BaseException | None:
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

    def workflowResult(self) -> bool | None:
        return self.__workflow_result

    def workflowReport(self) -> dict[str, dict[str, Any]]:
        return self.__status
