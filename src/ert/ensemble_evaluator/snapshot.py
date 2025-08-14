from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime
from typing import Any, TypeVar, cast, get_args

from typing_extensions import TypedDict

from _ert.events import (
    EESnapshot,
    EESnapshotUpdate,
    EnsembleCancelled,
    EnsembleEvent,
    EnsembleFailed,
    EnsembleStarted,
    EnsembleSucceeded,
    FMEvent,
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    RealizationEvent,
    RealizationFailed,
    RealizationPending,
    RealizationResubmit,
    RealizationRunning,
    RealizationStoppedLongRunning,
    RealizationSuccess,
    RealizationTimeout,
    RealizationUnknown,
    RealizationWaiting,
)
from _ert.forward_model_runner.fm_dispatch import FORWARD_MODEL_TERMINATED_MSG
from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator import state

logger = logging.getLogger(__name__)


class UnsupportedOperationException(ValueError):
    pass


_FM_TYPE_EVENT_TO_STATUS = {
    RealizationWaiting: state.REALIZATION_STATE_WAITING,
    RealizationPending: state.REALIZATION_STATE_PENDING,
    RealizationRunning: state.REALIZATION_STATE_RUNNING,
    RealizationFailed: state.REALIZATION_STATE_FAILED,
    RealizationSuccess: state.REALIZATION_STATE_FINISHED,
    RealizationUnknown: state.REALIZATION_STATE_UNKNOWN,
    RealizationTimeout: state.REALIZATION_STATE_FAILED,
    RealizationStoppedLongRunning: state.REALIZATION_STATE_FAILED,
    # For consistency since realization will turn to waiting state when resubmitted:
    RealizationResubmit: state.REALIZATION_STATE_WAITING,
    ForwardModelStepStart: state.FORWARD_MODEL_STATE_START,
    ForwardModelStepRunning: state.FORWARD_MODEL_STATE_RUNNING,
    ForwardModelStepSuccess: state.FORWARD_MODEL_STATE_FINISHED,
    ForwardModelStepFailure: state.FORWARD_MODEL_STATE_FAILURE,
}

_ENSEMBLE_TYPE_EVENT_TO_STATUS = {
    EnsembleStarted: state.ENSEMBLE_STATE_STARTED,
    EnsembleSucceeded: state.ENSEMBLE_STATE_STOPPED,
    EnsembleCancelled: state.ENSEMBLE_STATE_CANCELLED,
    EnsembleFailed: state.ENSEMBLE_STATE_FAILED,
}


def convert_iso8601_to_datetime(
    timestamp: datetime | str | None,
) -> datetime | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        return timestamp

    return datetime.fromisoformat(timestamp)


RealId = str
FmStepId = str


class EnsembleSnapshotMetadata(TypedDict):
    # contains state mapped to QColor used in the GUI for each fm_step
    fm_step_status: defaultdict[RealId, dict[FmStepId, str]]
    # contains state mapped to QColor used in the GUI for each real
    real_status: dict[RealId, str]
    sorted_real_ids: list[RealId]
    sorted_fm_step_ids: defaultdict[RealId, list[FmStepId]]


class EnsembleSnapshot:
    """The snapshot class is how we communicate the state of the ensemble between
    ensemble_evaluator and monitors. We start with an empty snapshot and as
    realizations progress, we send smaller snapshots only containing the changes
    which are then merged into the initial snapshot. In case a connection
    is dropped, we can send the entire snapshot.
    """

    def __init__(self) -> None:
        self._realization_snapshots: defaultdict[
            RealId,
            RealizationSnapshot,
        ] = defaultdict(RealizationSnapshot)

        self._fm_step_snapshots: defaultdict[
            tuple[RealId, FmStepId], FMStepSnapshot
        ] = defaultdict(FMStepSnapshot)

        self._ensemble_state: str | None = None
        self._metadata = EnsembleSnapshotMetadata(
            fm_step_status=defaultdict(dict),
            real_status={},
            sorted_real_ids=[],
            sorted_fm_step_ids=defaultdict(list),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnsembleSnapshot):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    @classmethod
    def from_nested_dict(cls, source: Mapping[Any, Any]) -> EnsembleSnapshot:
        ensemble = EnsembleSnapshot()
        if "metadata" in source:
            ensemble._metadata = source["metadata"]
        if "status" in source:
            ensemble._ensemble_state = source["status"]
        for real_id, realization_data in source.get("reals", {}).items():
            ensemble.add_realization(
                real_id, _realization_dict_to_realization_snapshot(realization_data)
            )
        return ensemble

    def add_realization(
        self, real_id: RealId, realization: RealizationSnapshot
    ) -> None:
        self._realization_snapshots[real_id] = realization

        for fm_step_id, fm_step_snapshot in realization.get("fm_steps", {}).items():
            fm_step_idx = (real_id, fm_step_id)
            self._fm_step_snapshots[fm_step_idx] = fm_step_snapshot

    def merge_snapshot(self, ensemble: EnsembleSnapshot) -> EnsembleSnapshot:
        self._metadata.update(ensemble._metadata)
        if ensemble._ensemble_state is not None:
            self._ensemble_state = ensemble._ensemble_state
        for real_id, other_real_data in ensemble._realization_snapshots.items():
            self._realization_snapshots[real_id].update(other_real_data)
        for fm_step_id, other_fm_data in ensemble._fm_step_snapshots.items():
            self._fm_step_snapshots[fm_step_id].update(other_fm_data)
        return self

    def merge_metadata(self, metadata: EnsembleSnapshotMetadata) -> None:
        self._metadata.update(metadata)

    def to_dict(self) -> dict[str, Any]:
        """used to send snapshot updates"""
        dict_: dict[str, Any] = {}
        if self._metadata:
            dict_["metadata"] = self._metadata
        if self._ensemble_state:
            dict_["status"] = self._ensemble_state
        if self._realization_snapshots:
            dict_["reals"] = {
                k: _filter_nones(v) for k, v in self._realization_snapshots.items()
            }

        for (real_id, fm_id), fm_values_dict in self._fm_step_snapshots.items():
            if "reals" not in dict_:
                dict_["reals"] = {}
            if real_id not in dict_["reals"]:
                dict_["reals"][real_id] = _filter_nones(
                    RealizationSnapshot(fm_steps={})
                )
            if "fm_steps" not in dict_["reals"][real_id]:
                dict_["reals"][real_id]["fm_steps"] = {}

            dict_["reals"][real_id]["fm_steps"][fm_id] = fm_values_dict

        return dict_

    @property
    def status(self) -> str | None:
        return self._ensemble_state

    @property
    def metadata(self) -> EnsembleSnapshotMetadata:
        return self._metadata

    def get_all_fm_steps(
        self,
    ) -> Mapping[tuple[RealId, FmStepId], FMStepSnapshot]:
        return self._fm_step_snapshots.copy()

    def get_fm_steps_for_all_reals(
        self,
    ) -> Mapping[tuple[RealId, FmStepId], str]:
        return {
            idx: fm_step_snapshot["status"]
            for idx, fm_step_snapshot in self._fm_step_snapshots.items()
            if "status" in fm_step_snapshot and fm_step_snapshot["status"] is not None
        }

    @property
    def reals(self) -> Mapping[RealId, RealizationSnapshot]:
        return self._realization_snapshots

    def get_fm_steps_for_real(self, real_id: RealId) -> dict[FmStepId, FMStepSnapshot]:
        return {
            fm_step_idx[1]: fm_step_snapshot.copy()
            for fm_step_idx, fm_step_snapshot in self._fm_step_snapshots.items()
            if fm_step_idx[0] == real_id
        }

    def get_real(self, real_id: RealId) -> RealizationSnapshot:
        return self._realization_snapshots[real_id]

    def get_fm_step(self, real_id: RealId, fm_step_id: FmStepId) -> FMStepSnapshot:
        return self._fm_step_snapshots[real_id, fm_step_id].copy()

    def get_successful_realizations(self) -> list[int]:
        return [
            int(real_idx)
            for real_idx, real_data in self._realization_snapshots.items()
            if real_data.get("status", "") == state.REALIZATION_STATE_FINISHED
        ]

    def aggregate_real_states(self) -> Counter[str]:
        counter = Counter(
            real["status"]
            for real in self._realization_snapshots.values()
            if real.get("status") is not None
        )
        return counter  # type: ignore

    def data(self) -> Mapping[str, Any]:
        # The gui uses this
        return self.to_dict()

    def update_realization(
        self,
        real_id: str,
        status: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        exec_hosts: str | None = None,
        message: str | None = None,
    ) -> EnsembleSnapshot:
        self._realization_snapshots[real_id].update(
            _filter_nones(
                RealizationSnapshot(
                    status=status,
                    start_time=start_time,
                    end_time=end_time,
                    exec_hosts=exec_hosts,
                    message=message,
                )
            )
        )
        return self

    def update_from_event(
        self,
        event: RealizationEvent
        | FMEvent
        | EnsembleEvent
        | EESnapshotUpdate
        | EESnapshot,
        source_snapshot: EnsembleSnapshot | None = None,
    ) -> EnsembleSnapshot:
        e_type = type(event)
        timestamp = event.time

        if source_snapshot is None:
            source_snapshot = EnsembleSnapshot()
        if e_type in get_args(RealizationEvent):
            event = cast(RealizationEvent, event)
            status = _FM_TYPE_EVENT_TO_STATUS[type(event)]
            start_time = None
            end_time = None
            exec_hosts = event.exec_hosts
            message = None

            if e_type is RealizationRunning:
                start_time = convert_iso8601_to_datetime(timestamp)
            elif e_type in {
                RealizationSuccess,
                RealizationFailed,
                RealizationTimeout,
                RealizationStoppedLongRunning,
            }:
                end_time = convert_iso8601_to_datetime(timestamp)
            if type(event) is RealizationFailed:
                message = event.message
            self.update_realization(
                event.real,
                status,
                start_time,
                end_time,
                exec_hosts,
                message,
            )

            if e_type is RealizationTimeout:
                for (
                    fm_step_id,
                    fm_step,
                ) in source_snapshot.get_fm_steps_for_real(event.real).items():
                    if fm_step.get(ids.STATUS) != state.FORWARD_MODEL_STATE_FINISHED:
                        fm_idx = (event.real, fm_step_id)
                        if fm_idx not in source_snapshot._fm_step_snapshots:
                            self._fm_step_snapshots[fm_idx] = FMStepSnapshot()
                        self._fm_step_snapshots[fm_idx].update(
                            FMStepSnapshot(
                                status=state.FORWARD_MODEL_STATE_FAILURE,
                                end_time=end_time,
                                error="The run is cancelled due to "
                                "reaching MAX_RUNTIME",
                            )
                        )
            elif e_type is RealizationStoppedLongRunning:
                for (
                    fm_step_id,
                    fm_step,
                ) in source_snapshot.get_fm_steps_for_real(event.real).items():
                    if fm_step.get(ids.STATUS) != state.FORWARD_MODEL_STATE_FINISHED:
                        fm_idx = (event.real, fm_step_id)
                        if fm_idx not in source_snapshot._fm_step_snapshots:
                            self._fm_step_snapshots[fm_idx] = FMStepSnapshot()
                        self._fm_step_snapshots[fm_idx].update(
                            FMStepSnapshot(
                                status=state.FORWARD_MODEL_STATE_FAILURE,
                                end_time=end_time,
                                error="The run is cancelled due to "
                                "excessive runtime, 25% more than the average "
                                "runtime (check keyword STOP_LONG_RUNNING)",
                            )
                        )
            elif e_type is RealizationResubmit:
                for fm_step_id in source_snapshot.get_fm_steps_for_real(event.real):
                    fm_idx = (event.real, fm_step_id)
                    self._fm_step_snapshots[fm_idx].update(
                        FMStepSnapshot(
                            status=state.FORWARD_MODEL_STATE_INIT,
                            start_time=None,
                            end_time=None,
                            current_memory_usage=None,
                            max_memory_usage=None,
                            cpu_seconds=None,
                            error=None,
                            stdout=None,
                            stderr=None,
                        )
                    )

        elif e_type in get_args(FMEvent):
            event = cast(FMEvent, event)
            status = _FM_TYPE_EVENT_TO_STATUS[type(event)]
            start_time = None
            end_time = None
            error = None
            if e_type is ForwardModelStepStart:
                start_time = convert_iso8601_to_datetime(timestamp)
            elif e_type in {ForwardModelStepSuccess, ForwardModelStepFailure}:
                end_time = convert_iso8601_to_datetime(timestamp)

                if type(event) is ForwardModelStepFailure:
                    error = event.error_msg or ""
                else:
                    # Make sure error msg from previous failed run is replaced
                    error = ""

            fm = _filter_nones(
                FMStepSnapshot(
                    status=status,
                    index=event.fm_step,
                    start_time=start_time,
                    end_time=end_time,
                    error=error,
                )
            )

            if type(event) is ForwardModelStepRunning:
                fm["current_memory_usage"] = event.current_memory_usage
                fm["max_memory_usage"] = event.max_memory_usage
                fm["cpu_seconds"] = event.cpu_seconds
            if type(event) is ForwardModelStepStart:
                fm["stdout"] = event.std_out
                fm["stderr"] = event.std_err

            self.update_fm_step(
                event.real,
                event.fm_step,
                fm,
            )

        elif e_type in get_args(EnsembleEvent):
            event = cast(EnsembleEvent, event)
            self._ensemble_state = _ENSEMBLE_TYPE_EVENT_TO_STATUS[type(event)]
        elif type(event) is EESnapshotUpdate:
            self.merge_snapshot(EnsembleSnapshot.from_nested_dict(event.snapshot))
        elif type(event) is EESnapshot:
            return EnsembleSnapshot.from_nested_dict(event.snapshot)
        else:
            raise ValueError(f"Unknown type: {e_type}")
        return self

    def update_fm_step(
        self,
        real_id: str,
        fm_step_id: str,
        fm_step: FMStepSnapshot,
    ) -> EnsembleSnapshot:
        if (
            previous_error := self._fm_step_snapshots[real_id, fm_step_id].get("error")
        ) and fm_step.get("error") == FORWARD_MODEL_TERMINATED_MSG:
            fm_step["error"] = previous_error
        self._fm_step_snapshots[real_id, fm_step_id].update(fm_step)
        return self


class FMStepSnapshot(TypedDict, total=False):
    status: str | None
    start_time: datetime | None
    end_time: datetime | None
    index: str | None
    current_memory_usage: int | None
    max_memory_usage: int | None
    cpu_seconds: float | None
    name: str | None
    error: str | None
    stdout: str | None
    stderr: str | None


class RealizationSnapshot(TypedDict, total=False):
    status: str | None
    active: bool | None
    start_time: datetime | None
    end_time: datetime | None
    exec_hosts: str | None
    fm_steps: dict[str, FMStepSnapshot]
    message: str | None


def _realization_dict_to_realization_snapshot(
    source: dict[str, Any],
) -> RealizationSnapshot:
    realization = RealizationSnapshot(
        status=source.get("status"),
        active=source.get("active"),
        start_time=convert_iso8601_to_datetime(source.get("start_time")),
        end_time=convert_iso8601_to_datetime(source.get("end_time")),
        exec_hosts=source.get("exec_hosts"),
        message=source.get("message"),
        fm_steps=source.get("fm_steps", {}),
    )
    for step in realization["fm_steps"].values():
        if step.get("start_time"):
            step["start_time"] = convert_iso8601_to_datetime(step["start_time"])
        if step.get("end_time"):
            step["end_time"] = convert_iso8601_to_datetime(step["end_time"])
    return _filter_nones(realization)


T = TypeVar("T", RealizationSnapshot, FMStepSnapshot)


def _filter_nones(data: T) -> T:
    return cast(T, {k: v for k, v in data.items() if v is not None})
