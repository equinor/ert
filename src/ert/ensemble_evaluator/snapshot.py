import datetime
import re
import typing
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from cloudevents.http import CloudEvent
from dateutil.parser import parse
from pydantic import BaseModel

from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator import state

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


def _filter_nones(some_dict: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in some_dict.items() if value is not None}


class PartialSnapshot:
    def __init__(self, snapshot: Optional["Snapshot"] = None) -> None:
        self._realization_states: Dict[
            str, Dict[str, Union[bool, datetime.datetime, str]]
        ] = defaultdict(dict)
        """A shallow dictionary of realization states. The key is a string with
        realization number, pointing to a dict with keys active (bool),
        start_time (datetime), end_time (datetime) and status (str)."""

        self._step_states: Dict[
            Tuple[str, str], Dict[str, Union[str, datetime.datetime]]
        ] = defaultdict(dict)
        """A shallow dictionary of step states. The key is a tuple of two strings with
        realization id and step id, pointing to a dict with the same members as the Step
        class, except Jobs"""

        self._job_states: Dict[
            Tuple[str, str, str], Dict[str, Union[str, datetime.datetime]]
        ] = defaultdict(dict)
        """A shallow dictionary of job states. The key is a tuple of three
        strings with realization id, step id and job id, pointing to a dict with
        the same members as the Job."""

        self._ensemble_state: Optional[str] = None
        # TODO not sure about possible values at this point, as GUI hijacks this one as
        # well
        self._metadata: Dict[str, Any] = defaultdict(dict)

        self._snapshot = snapshot

    @property
    def status(self) -> Optional[str]:
        return self._ensemble_state

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """only used in gui snapshot model, which only cares about the partial
        snapshot's metadata"""
        self._metadata.update(_filter_nones(metadata))

    def update_step(
        self, real_id: str, step_id: str, step: "Step"
    ) -> "PartialSnapshot":
        step_idx = (real_id, step_id)
        step_update = _filter_nones(
            {
                "status": step.status,
                "start_time": step.start_time,
                "end_time": step.end_time,
            }
        )
        self._step_states[step_idx].update(step_update)
        if self._snapshot:
            self._snapshot._my_partial._step_states[step_idx].update(step_update)
        self._check_state_after_step_update(step_idx[0], step_idx[1])
        return self

    def update_job(
        self,
        real_id: str,
        step_id: str,
        job_id: str,
        job: "Job",
    ) -> "PartialSnapshot":
        job_idx = (real_id, step_id, job_id)
        job_update = _filter_nones(job.dict())

        self._job_states[job_idx].update(job_update)
        if self._snapshot:
            self._snapshot._my_partial._job_states[job_idx].update(job_update)
        return self

    def _check_state_after_step_update(
        self, real_id: str, step_id: str
    ) -> "PartialSnapshot":
        step = self._step_states[(real_id, step_id)]
        step_status = step.get("status")
        assert isinstance(step_status, str)
        assert self._snapshot is not None

        real_state = self._realization_states[real_id]
        if real_state.get("status") == state.REALIZATION_STATE_FAILED:
            return self
        if step_status in _STEP_STATE_TO_REALIZATION_STATE:
            self._realization_states[real_id].update(
                {"status": _STEP_STATE_TO_REALIZATION_STATE[step_status]}
            )
        elif (
            step_status == state.REALIZATION_STATE_FINISHED
            and self._snapshot.all_steps_finished(real_id)
        ):
            real_state["status"] = state.REALIZATION_STATE_FINISHED
        elif (
            step_status == state.STEP_STATE_SUCCESS
            and not self._snapshot.all_steps_finished(real_id)
        ):
            pass
        else:
            raise ValueError(
                f"unknown step status {step_status} for real: {real_id} step: "
                + f"{step_id}"
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """used to send snapshot updates - for thread safety, this method should not
        access the _snapshot property"""
        _dict: Dict[str, Any] = {}
        if self._metadata:
            _dict["metadata"] = self._metadata
        if self._ensemble_state:
            _dict["status"] = self._ensemble_state
        if self._realization_states:
            _dict["reals"] = self._realization_states

        for step_index_tuple, step_state in self._step_states.items():
            real_id = step_index_tuple[0]
            step_id = step_index_tuple[1]
            if "reals" not in _dict:
                _dict["reals"] = {real_id: {}}
            if "steps" not in _dict["reals"][real_id]:
                _dict["reals"][real_id]["steps"] = {}
            if step_id not in _dict["reals"][real_id]["steps"]:
                _dict["reals"][real_id]["steps"][step_id] = step_state
                _dict["reals"][real_id]["steps"][step_id]["jobs"] = {}

        for job_tuple, job_values_dict in self._job_states.items():
            real_id = job_tuple[0]
            step_id = job_tuple[1]
            if "reals" not in _dict:
                _dict["reals"] = {}
            if real_id not in _dict["reals"]:
                _dict["reals"][real_id] = {}
            if "steps" not in _dict["reals"][real_id]:
                _dict["reals"][real_id]["steps"] = {}
            if step_id not in _dict["reals"][real_id]["steps"]:
                _dict["reals"][real_id]["steps"][step_id] = {"jobs": {}}

            job_id = job_tuple[2]
            _dict["reals"][real_id]["steps"][step_id]["jobs"][job_id] = job_values_dict

        return _dict

    def data(self) -> Mapping[str, Any]:
        return self.to_dict()

    def _merge(self, other: "PartialSnapshot") -> "PartialSnapshot":
        self._metadata.update(other._metadata)
        if other._ensemble_state is not None:
            self._ensemble_state = other._ensemble_state
        for real_id, other_real_data in other._realization_states.items():
            self._realization_states[real_id].update(other_real_data)
        for step_id, other_step_data in other._step_states.items():
            self._step_states[step_id].update(other_step_data)
        for job_id, other_job_data in other._job_states.items():
            self._job_states[job_id].update(other_job_data)
        return self

    # pylint: disable=too-many-branches
    def from_cloudevent(self, event: CloudEvent) -> "PartialSnapshot":
        # pylint: disable=too-many-statements
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
                Step(
                    **_filter_nones(
                        {
                            "status": status,
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                    )
                ),
            )

            if e_type == ids.EVTYPE_FM_STEP_TIMEOUT:
                step = self._snapshot.get_step(
                    _get_real_id(e_source), _get_step_id(e_source)
                )
                for job_id, job in step.jobs.items():
                    if job.status != state.JOB_STATE_FINISHED:
                        real_id = _get_real_id(e_source)
                        step_id = _get_step_id(e_source)
                        job_idx = (real_id, step_id, job_id)
                        if job_idx not in self._job_states:
                            self._job_states[job_idx] = {}
                        self._job_states[job_idx].update(
                            {
                                "status": state.JOB_STATE_FAILURE,
                                "end_time": end_time,  # type: ignore
                                "error": "The run is cancelled due to "
                                "reaching MAX_RUNTIME",
                            }
                        )

        elif e_type in ids.EVGROUP_FM_JOB:
            start_time = None
            end_time = None
            if e_type == ids.EVTYPE_FM_JOB_START:
                start_time = convert_iso8601_to_datetime(timestamp)
            elif e_type in {ids.EVTYPE_FM_JOB_SUCCESS, ids.EVTYPE_FM_JOB_FAILURE}:
                end_time = convert_iso8601_to_datetime(timestamp)

            job_dict = {
                "status": status,
                "start_time": start_time,
                "end_time": end_time,
                "index": _get_job_index(e_source),
            }
            if e_type == ids.EVTYPE_FM_JOB_RUNNING:
                job_dict[ids.CURRENT_MEMORY_USAGE] = event.data.get(
                    ids.CURRENT_MEMORY_USAGE
                )
                job_dict[ids.MAX_MEMORY_USAGE] = event.data.get(ids.MAX_MEMORY_USAGE)
            if e_type == ids.EVTYPE_FM_JOB_START:
                job_dict["stdout"] = event.data.get(ids.STDOUT)
                job_dict["stderr"] = event.data.get(ids.STDERR)
            if e_type == ids.EVTYPE_FM_JOB_FAILURE:
                job_dict["error"] = event.data.get(ids.ERROR_MSG)
            self.update_job(
                _get_real_id(e_source),
                _get_step_id(e_source),
                _get_job_id(e_source),
                Job(**job_dict),
            )

        elif e_type in ids.EVGROUP_ENSEMBLE:
            self._ensemble_state = _ENSEMBLE_TYPE_EVENT_TO_STATUS[e_type]
        elif e_type == ids.EVTYPE_EE_SNAPSHOT_UPDATE:
            other_partial = _from_nested_dict(event.data)
            self._merge(other_partial)
        else:
            raise ValueError(f"Unknown type: {e_type}")
        return self


class Snapshot:
    def __init__(self, input_dict: Mapping[str, Any]) -> None:
        self._my_partial = _from_nested_dict(input_dict)

    def merge_event(self, event: PartialSnapshot) -> None:
        self._my_partial._merge(event)

    def merge(self, update_as_nested_dict: Mapping[str, Any]) -> None:
        self._my_partial._merge(_from_nested_dict(update_as_nested_dict))

    def merge_metadata(self, metadata: Dict[str, Any]) -> None:
        self._my_partial._metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        return self._my_partial.to_dict()

    @property
    def status(self) -> Optional[str]:
        return self._my_partial._ensemble_state

    @property
    def reals(self) -> Dict[str, "RealizationSnapshot"]:
        return {
            real_id: RealizationSnapshot(**real_data)
            for real_id, real_data in self._my_partial._realization_states.items()
        }

    def steps(self, real_id: str) -> Dict[str, "Step"]:
        return {
            step_idx[1]: Step(**step_data)
            for step_idx, step_data in self._my_partial._step_states.items()
            if step_idx[0] == real_id
        }

    def jobs(self, real_id: str, step_id: str) -> Dict[str, "Job"]:
        return {
            job_idx[2]: Job(**job_data)
            for job_idx, job_data in self._my_partial._job_states.items()
            if job_idx[0] == real_id and job_idx[1] == step_id
        }

    def get_real(self, real_id: str) -> "RealizationSnapshot":
        return RealizationSnapshot(**self._my_partial._realization_states[real_id])

    def get_step(self, real_id: str, step_id: str) -> "Step":
        return Step(**self._my_partial._step_states[(real_id, step_id)])

    def get_job(self, real_id: str, step_id: str, job_id: str) -> "Job":
        return Job(**self._my_partial._job_states[(real_id, step_id, job_id)])

    def all_steps_finished(self, real_id: str) -> bool:
        return all(
            step["status"] == state.STEP_STATE_SUCCESS
            for real_step_id, step in self._my_partial._step_states.items()
            if real_step_id[0] == real_id
        )

    def get_successful_realizations(self) -> int:
        return len(
            [
                real_idx
                for real_idx, real_data in self._my_partial._realization_states.items()
                if real_data[ids.STATUS] == state.REALIZATION_STATE_FINISHED
            ]
        )

    def aggregate_real_states(self) -> typing.Dict[str, int]:
        states: Dict[str, int] = defaultdict(int)
        for real in self._my_partial._realization_states.values():
            status = real["status"]
            assert isinstance(status, str)
            states[status] += 1
        return states

    def data(self) -> Mapping[str, Any]:
        # The gui uses this
        return self._my_partial.to_dict()


class Job(BaseModel):
    status: Optional[str]
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    index: Optional[str]
    current_memory_usage: Optional[str]
    max_memory_usage: Optional[str]
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
        current_memory_usage: Optional[str] = None,
        max_memory_usage: Optional[str] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ) -> "SnapshotBuilder":
        step = self.steps[step_id]
        step.jobs[job_id] = Job(
            status=status,
            index=index,
            start_time=start_time,
            end_time=end_time,
            name=name,
            stdout=stdout,
            stderr=stderr,
            current_memory_usage=current_memory_usage,
            max_memory_usage=max_memory_usage,
        )
        return self


def _from_nested_dict(data: Mapping[str, Any]) -> PartialSnapshot:
    partial = PartialSnapshot()
    if "metadata" in data:
        partial._metadata = data["metadata"]
    if "status" in data:
        partial._ensemble_state = data["status"]
    for real_id, realization_data in data.get("reals", {}).items():
        partial._realization_states[real_id] = _filter_nones(
            {
                "status": realization_data.get("status"),
                "active": realization_data.get("active"),
                "start_time": realization_data.get("start_time"),
                "end_time": realization_data.get("end_time"),
            }
        )
        for step_id, step_data in data["reals"][real_id].get("steps", {}).items():
            partial._step_states[(real_id, step_id)] = _filter_nones(
                {
                    "status": step_data.get("status"),
                    "start_time": step_data.get("start_time"),
                    "end_time": step_data.get("end_time"),
                }
            )
            for job_id, job in step_data.get("jobs", {}).items():
                job_idx = (real_id, step_id, job_id)
                partial._job_states[job_idx] = job

    return partial
