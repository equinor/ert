from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from _ert.events import (
    FMEvent,
    ForwardModelStepFailure,
    ForwardModelStepSuccess,
    SnapshotInputEvent,
)
from ert.config import ForwardModelStep, QueueConfig, QueueSystem
from ert.run_arg import RunArg

from .snapshot import EnsembleSnapshot, FMStepSnapshot, RealizationSnapshot
from .state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STARTED,
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_UNKNOWN,
    FORWARD_MODEL_STATE_INIT,
    REALIZATION_STATE_WAITING,
)

logger = logging.getLogger(__name__)

_handle = Callable[..., Any]


class _EnsembleStateTracker:
    def __init__(self, state_: str = ENSEMBLE_STATE_UNKNOWN) -> None:
        self._state = state_
        self._handles: dict[str, _handle] = {}
        self._msg = "Illegal state transition from %s to %s"

        self.set_default_handles()

    def add_handle(self, state_: str, handle: _handle) -> None:
        self._handles[state_] = handle

    def _handle_unknown(self) -> None:
        if self._state != ENSEMBLE_STATE_UNKNOWN:
            logger.warning(self._msg, self._state, ENSEMBLE_STATE_UNKNOWN)
        self._state = ENSEMBLE_STATE_UNKNOWN

    def _handle_started(self) -> None:
        if self._state != ENSEMBLE_STATE_UNKNOWN:
            logger.warning(self._msg, self._state, ENSEMBLE_STATE_STARTED)
        self._state = ENSEMBLE_STATE_STARTED

    def _handle_failed(self) -> None:
        if self._state not in {
            ENSEMBLE_STATE_UNKNOWN,
            ENSEMBLE_STATE_STARTED,
        }:
            logger.warning(self._msg, self._state, ENSEMBLE_STATE_FAILED)
        self._state = ENSEMBLE_STATE_FAILED

    def _handle_stopped(self) -> None:
        if self._state != ENSEMBLE_STATE_STARTED:
            logger.warning(self._msg, self._state, ENSEMBLE_STATE_STOPPED)
        self._state = ENSEMBLE_STATE_STOPPED

    def _handle_cancelled(self) -> None:
        if self._state != ENSEMBLE_STATE_STARTED:
            logger.warning(self._msg, self._state, ENSEMBLE_STATE_CANCELLED)
        self._state = ENSEMBLE_STATE_CANCELLED

    def set_default_handles(self) -> None:
        self.add_handle(ENSEMBLE_STATE_UNKNOWN, self._handle_unknown)
        self.add_handle(ENSEMBLE_STATE_STARTED, self._handle_started)
        self.add_handle(ENSEMBLE_STATE_FAILED, self._handle_failed)
        self.add_handle(ENSEMBLE_STATE_STOPPED, self._handle_stopped)
        self.add_handle(ENSEMBLE_STATE_CANCELLED, self._handle_cancelled)

    def update_state(self, state_: str) -> str:
        if state_ not in self._handles:
            raise KeyError(f"Handle not defined for state {state_}")

        # Call the state handle mapped to the new state
        self._handles[state_]()

        return self._state


@dataclass
class LegacyEnsemble:
    reals: list[Realization]
    metadata: dict[str, Any]
    _queue_config: QueueConfig
    min_required_realizations: int
    id_: str

    def __post_init__(self) -> None:
        self.snapshot: EnsembleSnapshot = self._create_snapshot()
        self.status = self.snapshot.status
        if self.snapshot.status:
            self._status_tracker = _EnsembleStateTracker(self.snapshot.status)
        else:
            self._status_tracker = _EnsembleStateTracker()

    @property
    def active_reals(self) -> Sequence[Realization]:
        return list(filter(lambda real: real.active, self.reals))

    def _create_snapshot(self) -> EnsembleSnapshot:
        snapshot = EnsembleSnapshot()
        snapshot._ensemble_state = ENSEMBLE_STATE_UNKNOWN
        for real in self.active_reals:
            realization = RealizationSnapshot(
                active=True, status=REALIZATION_STATE_WAITING, fm_steps={}
            )
            for index, fm_step in enumerate(real.fm_steps):
                realization["fm_steps"][str(index)] = FMStepSnapshot(
                    status=FORWARD_MODEL_STATE_INIT,
                    index=str(index),
                    name=fm_step.name,
                )
            snapshot.add_realization(str(real.iens), realization)
        return snapshot

    def get_successful_realizations(self) -> list[int]:
        return self.snapshot.get_successful_realizations()

    def _log_completed_fm_step(
        self, event: FMEvent, step_snapshot: FMStepSnapshot | None
    ) -> None:
        if step_snapshot is None:
            logger.warning(f"Should log {event}, but there was no step_snapshot")
            return
        step_name = step_snapshot.get("name", "")
        start_time = step_snapshot.get("start_time")
        cpu_seconds = step_snapshot.get("cpu_seconds")
        current_memory_usage = step_snapshot.get("current_memory_usage")
        if start_time is not None and event.time is not None:
            walltime = (event.time - start_time).total_seconds()
        else:
            # We get here if the Running event is in the same event batch as
            # the Success event. That means that runtime is close to zero.
            walltime = 0

        if walltime > 120:
            logger.info(
                f"{event.event_type} {step_name} "
                f"{walltime=} "
                f"{cpu_seconds=} "
                f"{current_memory_usage=} "
                f"step_index={event.fm_step} "
                f"real={event.real}"
            )

    def update_snapshot(self, events: Sequence[SnapshotInputEvent]) -> EnsembleSnapshot:
        snapshot_mutate_event = EnsembleSnapshot()
        for event in events:
            snapshot_mutate_event = snapshot_mutate_event.update_from_event(
                event, source_snapshot=self.snapshot
            )
        self.snapshot.merge_snapshot(snapshot_mutate_event)
        if self.snapshot.status is not None and self.status != self.snapshot.status:
            self.status = self._status_tracker.update_state(self.snapshot.status)

        for event in events:
            if isinstance(event, ForwardModelStepSuccess | ForwardModelStepFailure):
                step = (
                    self.snapshot.reals[event.real]
                    .get("fm_steps", {})
                    .get(event.fm_step)
                )
                self._log_completed_fm_step(event, step)

        return snapshot_mutate_event

    @property
    def cancellable(self) -> bool:
        return True

    @property
    def queue_system(self) -> QueueSystem:
        return self._queue_config.queue_system


@dataclass
class Realization:
    iens: int
    fm_steps: Sequence[ForwardModelStep]
    active: bool
    max_runtime: int | None
    run_arg: RunArg
    num_cpu: int
    job_script: str
    realization_memory: int  # Memory to reserve/book, in bytes
