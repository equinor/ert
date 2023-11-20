"""
Module implementing a queue for managing external jobs.

"""
from __future__ import annotations

import asyncio
import datetime
import logging
import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from lxml import etree
from statemachine import State, StateMachine

from ert.constant_filenames import ERROR_file, STATUS_file
from ert.realization_state import RealizationState as RealizationStorageState

if TYPE_CHECKING:
    from ert.job_queue import JobQueue
    from ert.run_arg import RunArg


logger = logging.getLogger(__name__)


@dataclass
class QueueableRealization:  # Aka "Job" or previously "JobQueueNode"
    job_script: pathlib.Path
    run_arg: "RunArg"
    num_cpu: int = 1
    status_file: str = STATUS_file
    exit_file: str = ERROR_file
    max_runtime: Optional[int] = None
    callback_timeout: Optional[Callable[[int], None]] = None


class RealizationState(StateMachine):  # type: ignore
    NOT_ACTIVE = State("NOT ACTIVE", initial=True)
    WAITING = State("WAITING")
    SUBMITTED = State("SUBMITTED")
    PENDING = State("PENDING")
    RUNNING = State("RUNNING")
    DONE = State("DONE")
    EXIT = State("EXIT")
    DO_KILL = State("DO KILL")
    IS_KILLED = State("IS KILLED", final=True)
    SUCCESS = State("SUCCESS", final=True)
    STATUS_FAILURE = State("STATUS FAILURE")
    FAILED = State("FAILED", final=True)
    DO_KILL_NODE_FAILURE = State("DO KILL NODE FAILURE", final=True)
    UNKNOWN = State("UNKNOWN")

    def __init__(
        self, jobqueue: "JobQueue", realization: QueueableRealization, retries: int = 1
    ):
        self.jobqueue: "JobQueue" = (
            jobqueue  # For direct callbacks. Consider only supplying needed callbacks.
        )
        self.realization: QueueableRealization = realization
        self.iens: int = realization.run_arg.iens
        self.start_time: Optional[datetime.datetime] = None
        self.runtime: Optional[int] = None
        self.retries_left: int = retries
        self._max_submit = retries + 1
        self._callback_status_msg: Optional[str] = None
        super().__init__()

    allocate = UNKNOWN.to(NOT_ACTIVE)

    activate = NOT_ACTIVE.to(WAITING)
    submit = WAITING.to(SUBMITTED)  # from jobqueue
    accept = SUBMITTED.to(PENDING)  # from driver
    start = PENDING.to(RUNNING)  # from driver
    runend = RUNNING.to(DONE)  # from driver
    runfail = RUNNING.to(EXIT)  # from driver
    retry = EXIT.to(SUBMITTED)

    dokill = DO_KILL.from_(SUBMITTED, PENDING, RUNNING)
    remove = IS_KILLED.from_(NOT_ACTIVE, WAITING)

    verify_kill = DO_KILL.to(IS_KILLED)

    ack_killfailure = DO_KILL.to(DO_KILL_NODE_FAILURE)  # do we want to track this?

    # run_done_callback:
    validate = DONE.to(SUCCESS)
    invalidate = DONE.to(FAILED) | EXIT.to(FAILED)
    loadfailed = DONE.to(EXIT)

    somethingwentwrong = UNKNOWN.from_(
        NOT_ACTIVE,
        WAITING,
        SUBMITTED,
        PENDING,
        RUNNING,
        DONE,
        EXIT,
        DO_KILL,
    )

    donotgohere = UNKNOWN.to(STATUS_FAILURE)

    def on_enter_state(self, target: RealizationState) -> None:
        if self.jobqueue._statechanges_to_publish is None:
            return
        if target in (
            RealizationState.WAITING,
            RealizationState.PENDING,
            RealizationState.RUNNING,
            RealizationState.SUCCESS,
            RealizationState.FAILED,
            RealizationState.IS_KILLED,
        ):
            change = {self.realization.run_arg.iens: target.id}
            asyncio.create_task(self.jobqueue._statechanges_to_publish.put(change))

    def on_enter_SUBMITTED(self) -> None:
        asyncio.create_task(self.jobqueue.driver.submit(self))

    def on_enter_RUNNING(self) -> None:
        self.start_time = datetime.datetime.now()

    def on_enter_EXIT(self) -> None:
        self.realization.run_arg.ensemble_storage.state_map[
            self.realization.run_arg.iens
        ] = RealizationStorageState.LOAD_FAILURE

        if self.retries_left > 0:
            self.retry()
            self.retries_left -= 1
        else:
            logger.error(
                f"Realization: {self.realization.run_arg.iens} "
                f"failed after reaching max submit ({self._max_submit}):"
            )
            exit_file_path = (
                Path(self.realization.run_arg.runpath) / self.realization.exit_file
            )
            if exit_file_path.exists():
                exit_file = etree.parse(exit_file_path)
                failed_job = exit_file.find("job").text
                error_reason = exit_file.find("reason").text
                stderr_capture = exit_file.find("stderr").text

                stderr_file = ""
                if stderr_file_node := exit_file.find("stderr_file"):
                    stderr_file = stderr_file_node.text

                logger.error(
                    f"job {failed_job} failed with: '{error_reason}'\n"
                    f"\tstderr file: '{stderr_file}',\n"
                    f"\tits contents:{stderr_capture}"
                )
            self.invalidate()

    def on_enter_DONE(self) -> None:
        assert self.start_time
        self.runtime = int((datetime.datetime.now() - self.start_time).total_seconds())
        asyncio.create_task(self.jobqueue.run_done_callback(self))

    def on_enter_DO_KILL(self) -> None:
        asyncio.create_task(self.jobqueue.driver.kill(self))
