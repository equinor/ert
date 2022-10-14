import collections
import copy
import datetime
import re
import typing
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence, Union, cast

import pyrsistent
from cloudevents.http import CloudEvent
from dateutil.parser import parse
from pydantic import BaseModel
from pyrsistent import freeze
from pyrsistent.typing import PMap as TPMap

from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator import state


def _recursive_update(
    left: TPMap[str, Any],
    right: Union[Mapping[str, Any], TPMap[str, Any]],
    check_key: bool = True,
) -> TPMap[str, Any]:
    for k, v in right.items():
        if check_key and k not in left:
            raise ValueError(f"Illegal field {k}")
        if isinstance(v, collections.abc.Mapping):
            d_val = left.get(k)
            if not d_val:
                left = left.set(k, freeze(v))
            else:
                left = left.set(k, _recursive_update(d_val, v, check_key))
        else:
            left = left.set(k, v)
    return left


_regexp_pattern = r"(?<=/{token}/)[^/]+"


def _match_token(token: str, source: str) -> str:
    f_pattern = _regexp_pattern.format(token=token)
    match = re.search(f_pattern, source)
    return match if match is None else match.group()  # type: ignore


def _get_real_id(source: str) -> str:
    return _match_token("real", source)


def _get_step_id(source: str) -> str:
    return _match_token("step", source)


def _get_job_id(source: str) -> str:
    return _match_token("job", source)


def _get_job_index(source: str) -> str:
    return _match_token("index", source)


class UnsupportedOperationException(ValueError):
    pass


_FM_TYPE_EVENT_TO_STATUS = {
    ids.EVTYPE_FM_STEP_WAITING: state.STEP_STATE_WAITING,
    ids.EVTYPE_FM_STEP_PENDING: state.STEP_STATE_PENDING,
    ids.EVTYPE_FM_STEP_RUNNING: state.STEP_STATE_RUNNING,
    ids.EVTYPE_FM_STEP_FAILURE: state.STEP_STATE_FAILURE,
    ids.EVTYPE_FM_STEP_SUCCESS: state.STEP_STATE_SUCCESS,
    ids.EVTYPE_FM_STEP_UNKNOWN: state.STEP_STATE_UNKNOWN,
    ids.EVTYPE_FM_STEP_TIMEOUT: state.STEP_STATE_FAILURE,
    ids.EVTYPE_FM_JOB_START: state.JOB_STATE_START,
    ids.EVTYPE_FM_JOB_RUNNING: state.JOB_STATE_RUNNING,
    ids.EVTYPE_FM_JOB_SUCCESS: state.JOB_STATE_FINISHED,
    ids.EVTYPE_FM_JOB_FAILURE: state.JOB_STATE_FAILURE,
}

_ENSEMBLE_TYPE_EVENT_TO_STATUS = {
    ids.EVTYPE_ENSEMBLE_STARTED: state.ENSEMBLE_STATE_STARTED,
    ids.EVTYPE_ENSEMBLE_STOPPED: state.ENSEMBLE_STATE_STOPPED,
    ids.EVTYPE_ENSEMBLE_CANCELLED: state.ENSEMBLE_STATE_CANCELLED,
    ids.EVTYPE_ENSEMBLE_FAILED: state.ENSEMBLE_STATE_FAILED,
}

_STEP_STATE_TO_REALIZATION_STATE = {
    state.STEP_STATE_WAITING: state.REALIZATION_STATE_WAITING,
    state.STEP_STATE_PENDING: state.REALIZATION_STATE_PENDING,
    state.STEP_STATE_RUNNING: state.REALIZATION_STATE_RUNNING,
    state.STEP_STATE_UNKNOWN: state.REALIZATION_STATE_UNKNOWN,
    state.STEP_STATE_FAILURE: state.REALIZATION_STATE_FAILED,
}


def convert_iso8601_to_datetime(
    timestamp: Union[datetime.datetime, str]
) -> datetime.datetime:
    if isinstance(timestamp, datetime.datetime):
        return timestamp

    return parse(timestamp)


class PartialSnapshot:
    def __init__(self, snapshot: "Snapshot") -> None:
        """Create a PartialSnapshot. If no snapshot is provided, the object is
        a immutable POD, and any attempt at mutating it will raise an
        UnsupportedOperationException."""
        self._data: TPMap[str, Any] = pyrsistent.m()
        self._snapshot = copy.copy(snapshot) if snapshot else None

    def update_status(self, status: str) -> None:
        self._apply_update(SnapshotDict(status=status))

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        if self._snapshot is None:
            raise UnsupportedOperationException(
                f"updating metadata on {self.__class__} without providing a snapshot"
                + " is not supported"
            )
        dictionary = pyrsistent.pmap({ids.METADATA: metadata})
        self._data = _recursive_update(self._data, dictionary, check_key=False)
        self._snapshot.merge_metadata(metadata)

    def update_real(
        self,
        real_id: str,
        real: "RealizationSnapshot",
    ) -> None:
        self._apply_update(SnapshotDict(reals={real_id: real}))

    def _apply_update(self, update: "SnapshotDict") -> None:
        if self._snapshot is None:
            raise UnsupportedOperationException(
                f"trying to mutate {self.__class__} without providing a snapshot is "
                + "not supported"
            )
        dictionary = update.dict(
            exclude_unset=True, exclude_none=True, exclude_defaults=True
        )
        self._data = _recursive_update(self._data, dictionary, check_key=False)
        self._snapshot.merge(dictionary)

    def update_step(
        self, real_id: str, step_id: str, step: "Step"
    ) -> "PartialSnapshot":
        if self._snapshot is None:
            raise UnsupportedOperationException(
                f"cannot update step for {self.__class__} without providing a snapshot"
            )
        self._apply_update(
            SnapshotDict(reals={real_id: RealizationSnapshot(steps={step_id: step})})
        )
        if self._snapshot.get_real(real_id).status != state.REALIZATION_STATE_FAILED:
            if step.status in _STEP_STATE_TO_REALIZATION_STATE:
                self.update_real(
                    real_id,
                    RealizationSnapshot(
                        status=_STEP_STATE_TO_REALIZATION_STATE[step.status]
                    ),
                )
            elif (
                step.status == state.REALIZATION_STATE_FINISHED
                and self._snapshot.all_steps_finished(real_id)
            ):
                self.update_real(
                    real_id,
                    RealizationSnapshot(status=state.REALIZATION_STATE_FINISHED),
                )
            elif (
                step.status == state.STEP_STATE_SUCCESS
                and not self._snapshot.all_steps_finished(real_id)
            ):
                pass
            else:
                raise ValueError(
                    f"unknown step status {step.status} for real: {real_id} step: "
                    + f"{step_id}"
                )
        return self

    def update_job(
        self,
        real_id: str,
        step_id: str,
        job_id: str,
        job: "Job",
    ) -> "PartialSnapshot":
        self._apply_update(
            SnapshotDict(
                reals={
                    real_id: RealizationSnapshot(
                        steps={step_id: Step(jobs={job_id: job})}
                    )
                }
            )
        )
        return self

    def to_dict(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], pyrsistent.thaw(self._data))

    def data(self) -> TPMap[str, Any]:
        return self._data

    # pylint: disable=too-many-branches
    def from_cloudevent(self, event: CloudEvent) -> "PartialSnapshot":
        e_type = event["type"]
        e_source = event["source"]
        status = _FM_TYPE_EVENT_TO_STATUS.get(e_type)
        timestamp = event["time"]

        if self._snapshot is None:
            raise UnsupportedOperationException(
                f"updating {self.__class__} without a snapshot is not supported"
            )

        if e_type in ids.EVGROUP_FM_STEP:
            start_time = None
            end_time = None
            if e_type == ids.EVTYPE_FM_STEP_RUNNING:
                start_time = convert_iso8601_to_datetime(timestamp)
            elif e_type in {
                ids.EVTYPE_FM_STEP_SUCCESS,
                ids.EVTYPE_FM_STEP_FAILURE,
                ids.EVTYPE_FM_STEP_TIMEOUT,
            }:
                end_time = convert_iso8601_to_datetime(timestamp)

            self.update_step(
                _get_real_id(e_source),
                _get_step_id(e_source),
                step=Step(
                    status=status,
                    start_time=start_time,
                    end_time=end_time,
                ),
            )

            if e_type == ids.EVTYPE_FM_STEP_TIMEOUT:
                step = self._snapshot.get_step(
                    _get_real_id(e_source), _get_step_id(e_source)
                )
                for job_id, job in step.jobs.items():
                    if job.status != state.JOB_STATE_FINISHED:
                        job_error = "The run is cancelled due to reaching MAX_RUNTIME"
                        job_index = _get_job_index(e_source)
                        self.update_job(
                            _get_real_id(e_source),
                            _get_step_id(e_source),
                            job_id,
                            job=Job(
                                status=state.JOB_STATE_FAILURE,
                                index=job_index,
                                error=job_error,
                            ),
                        )

        elif e_type in ids.EVGROUP_FM_JOB:
            start_time = None
            end_time = None
            if e_type == ids.EVTYPE_FM_JOB_START:
                start_time = convert_iso8601_to_datetime(timestamp)
            elif e_type in {ids.EVTYPE_FM_JOB_SUCCESS, ids.EVTYPE_FM_JOB_FAILURE}:
                end_time = convert_iso8601_to_datetime(timestamp)
            job_index = _get_job_index(e_source)
            self.update_job(
                _get_real_id(e_source),
                _get_step_id(e_source),
                _get_job_id(e_source),
                job=Job(
                    status=status,
                    start_time=start_time,
                    end_time=end_time,
                    index=job_index,
                    data=event.data if e_type == ids.EVTYPE_FM_JOB_RUNNING else None,
                    stdout=event.data.get(ids.STDOUT)
                    if e_type == ids.EVTYPE_FM_JOB_START
                    else None,
                    stderr=event.data.get(ids.STDERR)
                    if e_type == ids.EVTYPE_FM_JOB_START
                    else None,
                    error=event.data.get(ids.ERROR_MSG)
                    if e_type == ids.EVTYPE_FM_JOB_FAILURE
                    else None,
                ),
            )
        elif e_type in ids.EVGROUP_ENSEMBLE:
            self.update_status(_ENSEMBLE_TYPE_EVENT_TO_STATUS[e_type])
        elif e_type == ids.EVTYPE_EE_SNAPSHOT_UPDATE:
            self._data = _recursive_update(self._data, event.data, check_key=False)
        else:
            raise ValueError(f"Unknown type: {e_type}")
        return self


class Snapshot:
    def __init__(self, input_dict: Mapping[str, Any]) -> None:
        self._data: TPMap[str, Any] = pyrsistent.freeze(input_dict)

    def merge_event(self, event: PartialSnapshot) -> None:
        self._data = _recursive_update(self._data, event.data())

    def merge(self, update: Mapping[str, Any]) -> None:
        self._data = _recursive_update(self._data, update)

    def merge_metadata(self, metadata: Dict[str, Any]) -> None:
        self._data = _recursive_update(
            self._data, pyrsistent.pmap({ids.METADATA: metadata}), check_key=False
        )

    def to_dict(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], pyrsistent.thaw(self._data))

    @property
    def status(self) -> str:
        return cast(str, self._data[ids.STATUS])

    @property
    def reals(self) -> Dict[str, "RealizationSnapshot"]:
        return SnapshotDict(**self._data).reals

    def get_real(self, real_id: str) -> "RealizationSnapshot":
        if real_id not in self._data[ids.REALS]:
            raise ValueError(f"No realization with id {real_id}")
        return RealizationSnapshot(**self._data[ids.REALS][real_id])

    def get_step(self, real_id: str, step_id: str) -> "Step":
        real = self.get_real(real_id)
        steps = real.steps
        if step_id not in steps:
            raise ValueError(f"No step with id {step_id} in {real_id}")
        return steps[step_id]

    def get_job(self, real_id: str, step_id: str, job_id: str) -> "Job":
        step = self.get_step(real_id, step_id)
        jobs = step.jobs
        if job_id not in jobs:
            raise ValueError(f"No job with id {job_id} in {step_id}")
        return jobs[job_id]

    def all_steps_finished(self, real_id: str) -> bool:
        real = self.get_real(real_id)
        return all(
            step.status == state.STEP_STATE_SUCCESS for step in real.steps.values()
        )

    def get_successful_realizations(self) -> int:
        return len(
            [
                real
                for real in self._data[ids.REALS].values()
                if real[ids.STATUS] == state.REALIZATION_STATE_FINISHED
            ]
        )

    def aggregate_real_states(self) -> typing.Dict[str, int]:
        states: Dict[str, int] = defaultdict(int)
        for real in self._data[ids.REALS].values():
            states[real[ids.STATUS]] += 1
        return states

    def data(self) -> Mapping[str, Any]:
        return self._data


class Job(BaseModel):
    status: Optional[str]
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    index: Optional[str]
    data: Optional[Dict[str, Any]]
    name: Optional[str]
    error: Optional[str]
    stdout: Optional[str]
    stderr: Optional[str]


class Step(BaseModel):
    status: Optional[str]
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    jobs: Dict[str, Job] = {}


class RealizationSnapshot(BaseModel):
    status: Optional[str]
    active: Optional[bool]
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    steps: Dict[str, Step] = {}


class SnapshotDict(BaseModel):
    status: Optional[str] = state.ENSEMBLE_STATE_UNKNOWN
    reals: Dict[str, RealizationSnapshot] = {}
    metadata: Dict[str, Any] = {}


class SnapshotBuilder(BaseModel):
    steps: Dict[str, Step] = {}
    metadata: Dict[str, Any] = {}

    def build(
        self,
        real_ids: Sequence[str],
        status: Optional[str],
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> Snapshot:
        top = SnapshotDict(status=status, metadata=self.metadata)
        for r_id in real_ids:
            top.reals[r_id] = RealizationSnapshot(
                active=True,
                steps=self.steps,
                start_time=start_time,
                end_time=end_time,
                status=status,
            )
        return Snapshot(top.dict())

    def add_step(
        self,
        step_id: str,
        status: Optional[str],
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> "SnapshotBuilder":
        self.steps[step_id] = Step(
            status=status, start_time=start_time, end_time=end_time
        )
        return self

    def add_job(  # pylint: disable=too-many-arguments
        self,
        step_id: str,
        job_id: str,
        index: str,
        name: Optional[str],
        status: Optional[str],
        data: Optional[Dict[str, Any]],
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ) -> "SnapshotBuilder":
        step = self.steps[step_id]
        step.jobs[job_id] = Job(
            status=status,
            index=index,
            data=data,
            start_time=start_time,
            end_time=end_time,
            name=name,
            stdout=stdout,
            stderr=stderr,
        )
        return self

    def add_metadata(self, key: str, value: Any) -> "SnapshotBuilder":
        self.metadata[key] = value
        return self
