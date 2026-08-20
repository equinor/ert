from __future__ import annotations

import datetime
import logging
import threading
import types
from concurrent import futures
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

from ert import ErtScript
from ert.config import (
    BaseErtScriptWorkflow,
    ErtScriptWorkflow,
    ExternalErtScript,
    Workflow,
    WorkflowFixtures,
    WorkflowJob,
)


class WorkflowJobStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowJobResult:
    name: str
    index: int
    arguments: list[str]
    stdout: str
    stderr: str
    status: WorkflowJobStatus
    timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC)
    )


class WorkflowJobRunner:
    def __init__(self, workflow_job: WorkflowJob) -> None:
        self.job = workflow_job
        self.__running = False
        self.__script: ErtScript | None = None
        self.__cancel_requested = False
        self._lock = threading.Lock()
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
        try:
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

            with self._lock:
                if isinstance(self.job, BaseErtScriptWorkflow):
                    ert_script_class = self.job.load_ert_script_class()
                    self.__script = ert_script_class()
                    # We let stop on fail either from class or config take
                    # precedence
                    self.stop_on_fail = (
                        self.job.stop_on_fail or self.__script.stop_on_fail
                    )

                else:
                    self.__script = ExternalErtScript(
                        self.job.executable,  # type: ignore
                    )
                    self.stop_on_fail = self.job.stop_on_fail

                # A cancellation requested before the script existed would
                # be lost; apply it once there is a script to cancel.
                if self.__cancel_requested:
                    self.__script.cancel()

            return self.__script.initializeAndRun(
                self.job.argument_types(), arguments, fixtures
            )
        finally:
            self.__running = False

    @property
    def name(self) -> str:
        return self.job.name

    @property
    def execution_type(self) -> str:
        if isinstance(self.job, ErtScriptWorkflow):
            return "internal python"
        return "external"

    def cancel(self) -> None:
        with self._lock:
            self.__cancel_requested = True
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
        hook: str | None = None,
    ) -> None:
        self.__workflow = workflow
        self.fixtures = fixtures
        self._hook = hook

        self.__workflow_result: bool | None = None
        self._workflow_executor = futures.ThreadPoolExecutor(max_workers=1)
        self._workflow_job: Future[None] | None = None

        self.__running = False
        self.__cancelled = False
        self.__current_job: WorkflowJobRunner | None = None
        self.__status: dict[str, dict[str, Any]] = {}
        self.__job_results: list[WorkflowJobResult] = []
        self._current_job_lock = threading.Lock()

    def __enter__(self) -> Self:
        self.run()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
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
        self.__job_results = []
        self.__running = True

        for index, (job, args) in enumerate(self.__workflow):
            with self._current_job_lock:
                if self.__cancelled:
                    # The workflow was cancelled before this job started
                    result = WorkflowJobResult(
                        name=job.name,
                        index=index,
                        arguments=[str(arg) for arg in args],
                        stdout="",
                        stderr="",
                        status=WorkflowJobStatus.CANCELLED,
                    )
                    self.__job_results.append(result)
                    logger.info(self._log_entry(result), extra=self._log_extra(result))
                    continue

                jobrunner = WorkflowJobRunner(job)
                self.__current_job = jobrunner

            logger.info(
                f"Workflow job starting; {self._job_description(jobrunner.name, index)}"
            )
            jobrunner.run(args, fixtures=self.fixtures)

            if self.__cancelled:
                status = WorkflowJobStatus.CANCELLED
            elif jobrunner.hasFailed():
                status = WorkflowJobStatus.FAILED
            else:
                status = WorkflowJobStatus.SUCCESS

            self.__status[jobrunner.name] = {
                "stdout": jobrunner.stdoutdata(),
                "stderr": jobrunner.stderrdata(),
                "completed": status is WorkflowJobStatus.SUCCESS,
            }
            result = WorkflowJobResult(
                name=jobrunner.name,
                index=index,
                arguments=[str(arg) for arg in args],
                stdout=jobrunner.stdoutdata(),
                stderr=jobrunner.stderrdata(),
                status=status,
            )
            self.__job_results.append(result)

            extra = self._log_extra(result, execution_type=jobrunner.execution_type)
            if status is WorkflowJobStatus.FAILED:
                logger.error(self._log_entry(result), extra=extra)
            else:
                logger.info(self._log_entry(result), extra=extra)

            if jobrunner.hasFailed() and jobrunner.stop_on_fail:
                self.__running = False
                raise RuntimeError(
                    f"Workflow job {result.name} failed with error: {result.stderr}"
                )

        self.__current_job = None
        self.__running = False
        self.__workflow_result = True

    def _job_description(self, job_name: str, index: int) -> str:
        """Identify a job invocation the same way in every workflow log line."""
        return (
            f"hook={self._hook} workflow={self.__workflow.name} job={job_name}#{index}"
        )

    def _log_entry(self, result: WorkflowJobResult) -> str:
        description = self._job_description(result.name, result.index)
        sections = [f"Workflow job result; {description} status={result.status}"]
        if result.arguments:
            sections.append(f"--- arguments ---\n{' '.join(result.arguments)}")
        if result.stdout:
            sections.append(f"--- stdout ---\n{result.stdout.rstrip('\n')}")
        if result.stderr:
            sections.append(f"--- stderr ---\n{result.stderr.rstrip('\n')}")
        return "\n".join(sections)

    def _log_extra(
        self, result: WorkflowJobResult, execution_type: str | None = None
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "class": "WORKFLOW_JOB",
            "job_name": result.name,
            "workflow_name": self.__workflow.name,
            "arguments": " ".join(result.arguments),
            "status": result.status,
        }
        if self._hook is not None:
            extra["hook"] = self._hook
        if execution_type is not None:
            extra["execution_type"] = execution_type
        return extra

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
        with self._current_job_lock:
            self.__cancelled = True
            current_job = self.__current_job
        if current_job is not None:
            current_job.cancel()

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

    def workflow_job_results(self) -> list[WorkflowJobResult]:
        """One entry per job invocation, in the order the jobs were run.

        Unlike workflowReport(), which is keyed by job name, this keeps the
        output of every invocation when the same job is run more than once.
        """
        return self.__job_results
