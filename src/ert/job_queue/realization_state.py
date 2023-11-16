"""
Module implementing a queue for managing external jobs.

"""
from __future__ import annotations

import asyncio
import datetime
import logging
import pathlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional

from statemachine import StateMachine, states

from ert.constant_filenames import ERROR_file, STATUS_file

if TYPE_CHECKING:
    from ert.job_queue import JobQueue
    from ert.run_arg import RunArg


logger = logging.getLogger(__name__)


class JobStatus(Enum):
    # This value is used in external query routines - for jobs which are
    # (currently) not active.
    NOT_ACTIVE = auto()

    WAITING = auto()  # A node which is waiting in the internal queue.

    # (should) place it as pending or running.
    SUBMITTED = auto()

    # A node which is pending - a status returned by the external system. I.e LSF
    PENDING = auto()

    RUNNING = auto()  # The job is running

    # The job is done - but we have not yet checked if the target file is
    # produced:
    DONE = auto()

    # The job has exited - check attempts to determine if we retry or go to
    # complete_fail
    EXIT = auto()

    # The the job should be killed, either due to user request, or automated
    # measures - the job can NOT be restarted..
    DO_KILL = auto()

    # The job has been killed, following a DO_KILL - can restart.
    IS_KILLED = auto()

    # Validation went fine:
    SUCCESS = auto()

    STATUS_FAILURE = auto()  # Temporary failure, should not be a reachable state

    FAILED = auto()  # No more retries
    DO_KILL_NODE_FAILURE = auto()  # Compute node should be blocked
    UNKNOWN = auto()

    def __str__(self):
        return super().__str__().replace("JobStatus.", "")


@dataclass
class QueueableRealization:  # Aka "Job" or previously "JobQueueNode"
    job_script: pathlib.Path
    run_arg: "RunArg"
    num_cpu: int = 1
    status_file: str = STATUS_file
    exit_file: str = ERROR_file
    max_runtime: Optional[int] = None
    callback_timeout: Optional[Callable[[int], None]] = None

    def __hash__(self):
        # Elevate iens up to two levels? Check if it can be removed from run_arg
        return self.run_arg.iens

    def __repr__(self):
        return str(self.run_arg.iens)


class RealizationState(StateMachine):
    def __init__(
        self, jobqueue: "JobQueue", realization: QueueableRealization, retries: int = 1
    ):
        self.jobqueue: "JobQueue" = (
            jobqueue  # For direct callbacks. Consider only supplying needed callbacks.
        )
        self.realization: QueueableRealization = realization
        self.iens: int = realization.run_arg.iens
        self.start_time: datetime.datetime = (
            0  # When this realization moved into RUNNING (datetime?)
        )
        self.retries_left: int = retries
        super().__init__()

    _ = states.States.from_enum(
        JobStatus,
        initial=JobStatus.WAITING,
        final={
            JobStatus.SUCCESS,
            JobStatus.FAILED,
            JobStatus.IS_KILLED,
            JobStatus.DO_KILL_NODE_FAILURE,
        },
    )

    allocate = _.UNKNOWN.to(_.NOT_ACTIVE)

    activate = _.NOT_ACTIVE.to(_.WAITING)
    submit = _.WAITING.to(_.SUBMITTED)  # from jobqueue
    accept = _.SUBMITTED.to(_.PENDING)  # from driver
    start = _.PENDING.to(_.RUNNING)  # from driver
    runend = _.RUNNING.to(_.DONE)  # from driver
    runfail = _.RUNNING.to(_.EXIT)  # from driver
    retry = _.EXIT.to(_.SUBMITTED)

    dokill = _.DO_KILL.from_(_.SUBMITTED, _.PENDING, _.RUNNING)

    verify_kill = _.DO_KILL.to(_.IS_KILLED)

    ack_killfailure = _.DO_KILL.to(_.DO_KILL_NODE_FAILURE)  # do we want to track this?

    validate = _.DONE.to(_.SUCCESS)
    invalidate = _.DONE.to(_.FAILED)

    somethingwentwrong = _.UNKNOWN.from_(
        _.NOT_ACTIVE,
        _.WAITING,
        _.SUBMITTED,
        _.PENDING,
        _.RUNNING,
        _.DONE,
        _.EXIT,
        _.DO_KILL,
    )

    donotgohere = _.UNKNOWN.to(_.STATUS_FAILURE)

    def on_enter_state(self, target, event):
        if target in (
            # RealizationState.WAITING,  # This happens too soon (initially)
            RealizationState.PENDING,
            RealizationState.RUNNING,
            RealizationState.SUCCESS,
            RealizationState.FAILED,
        ):
            change = {self.realization.run_arg.iens: target.id}
            asyncio.create_task(self.jobqueue._changes_to_publish.put(change))

    def on_enter_SUBMITTED(self):
        asyncio.create_task(self.jobqueue.driver.submit(self))

    def on_enter_RUNNING(self):
        self.start_time = datetime.datetime.now()

    def on_enter_EXIT(self):
        if self.retries_left > 0:
            self.retry()  # I think this adds to an "event queue" for the statemachine, if not, wrap it in an async task?
            self.retries_left -= 1
        else:
            self.invalidate()

    def on_enter_DONE(self):
        asyncio.create_task(self.jobqueue.run_done_callback(self))

    def on_enter_DO_KILL(self):
        asyncio.create_task(self.jobqueue.driver.kill(self))
