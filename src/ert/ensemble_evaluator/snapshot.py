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


def _get_job_id(source: str) -> str:
    return _match_token("job", source)


def _get_job_index(source: str) -> str:
    return _match_token("index", source)


class UnsupportedOperationException(ValueError):
    pass


_FM_TYPE_EVENT_TO_STATUS = {
    ids.EVTYPE_REALIZATION_WAITING: state.REALIZATION_STATE_WAITING,
    ids.EVTYPE_REALIZATION_PENDING: state.REALIZATION_STATE_PENDING,
    ids.EVTYPE_REALIZATION_RUNNING: state.REALIZATION_STATE_RUNNING,
    ids.EVTYPE_REALIZATION_FAILURE: state.REALIZATION_STATE_FAILED,
    ids.EVTYPE_REALIZATION_SUCCESS: state.REALIZATION_STATE_FINISHED,
    ids.EVTYPE_REALIZATION_UNKNOWN: state.REALIZATION_STATE_UNKNOWN,
    ids.EVTYPE_REALIZATION_TIMEOUT: state.REALIZATION_STATE_FAILED,
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

        self._job_states: Dict[
            Tuple[str, str], Dict[str, Union[str, datetime.datetime]]
        ] = defaultdict(dict)
        """A shallow dictionary of job states. The key is a tuple of two
        strings with realization id and job id, pointing to a dict with
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

    def update_realization(
        self,
        real_id: str,
        status: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> "PartialSnapshot":
        self._realization_states[real_id].update(
            _filter_nones(
                {"status": status, "start_time": start_time, "end_time": end_time}
            )
        )
        return self

    def update_job(
        self,
        real_id: str,
        job_id: str,
        job: "Job",
    ) -> "PartialSnapshot":
        job_update = _filter_nones(job.dict())

        self._job_states[(real_id, job_id)].update(job_update)
        if self._snapshot:
            self._snapshot._my_partial._job_states[(real_id, job_id)].update(job_update)
        return self

    def get_all_jobs(
        self,
    ) -> Mapping[Tuple[str, str], "Job"]:
        if self._snapshot:
            return self._snapshot.get_all_jobs()
        return {}

    def get_job_status_for_all_reals(
        self,
    ) -> Mapping[Tuple[str, str], Union[str, datetime.datetime]]:
        if self._snapshot:
            return self._snapshot.get_job_status_for_all_reals()
        return {}

    @property
    def reals(self) -> Mapping[str, "RealizationSnapshot"]:
        return {
            real_id: RealizationSnapshot(**real_data)
            for real_id, real_data in self._realization_states.items()
        }

    def get_real_ids(self) -> Sequence[str]:
        """we can have information about realizations in both _realization_states and
        _job_states - we combine the existing IDs"""
        real_ids = []
        for idx in self._job_states:
            real_id = idx[0]
            if real_id not in real_ids:
                real_ids.append(real_id)
        for real_id in self._realization_states:
            if real_id not in real_ids:
                real_ids.append(real_id)
        return sorted(real_ids, key=int)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def get_real(self, real_id: str) -> "RealizationSnapshot":
        return RealizationSnapshot(**self._realization_states[real_id])

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

        for job_tuple, job_values_dict in self._job_states.items():
            real_id = job_tuple[0]
            if "reals" not in _dict:
                _dict["reals"] = {}
            if real_id not in _dict["reals"]:
                _dict["reals"][real_id] = {}
            if "jobs" not in _dict["reals"][real_id]:
                _dict["reals"][real_id]["jobs"] = {}

            job_id = job_tuple[1]
            _dict["reals"][real_id]["jobs"][job_id] = job_values_dict

        return _dict

    def data(self) -> Mapping[str, Any]:
        return self.to_dict()

    def _merge(self, other: "PartialSnapshot") -> "PartialSnapshot":
        self._metadata.update(other._metadata)
        if other._ensemble_state is not None:
            self._ensemble_state = other._ensemble_state
        for real_id, other_real_data in other._realization_states.items():
            self._realization_states[real_id].update(other_real_data)
        for job_id, other_job_data in other._job_states.items():
            self._job_states[job_id].update(other_job_data)
        return self

    def from_cloudevent(self, event: CloudEvent) -> "PartialSnapshot":
        e_type = event["type"]
        e_source = event["source"]
        timestamp = event["time"]

        if self._snapshot is None:
            raise UnsupportedOperationException(
                f"updating {self.__class__} without a snapshot is not supported"
            )

        if e_type in ids.EVGROUP_REALIZATION:
            status = _FM_TYPE_EVENT_TO_STATUS[e_type]
            start_time = None
            end_time = None
            if e_type == ids.EVTYPE_REALIZATION_RUNNING:
                start_time = convert_iso8601_to_datetime(timestamp)
            elif e_type in {
                ids.EVTYPE_REALIZATION_SUCCESS,
                ids.EVTYPE_REALIZATION_FAILURE,
                ids.EVTYPE_REALIZATION_TIMEOUT,
            }:
                end_time = convert_iso8601_to_datetime(timestamp)

            self.update_realization(
                _get_real_id(e_source), status, start_time, end_time
            )

            if e_type == ids.EVTYPE_REALIZATION_TIMEOUT:
                for job_id, job in self._snapshot.get_jobs_for_real(
                    _get_real_id(e_source)
                ).items():
                    if job.status != state.JOB_STATE_FINISHED:
                        real_id = _get_real_id(e_source)
                        job_idx = (real_id, job_id)
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
            status = _FM_TYPE_EVENT_TO_STATUS[e_type]
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
    def metadata(self) -> Mapping[str, Any]:
        return self._my_partial.metadata

    def get_all_jobs(
        self,
    ) -> Mapping[Tuple[str, str], "Job"]:
        return {
            idx: Job(**job_state)
            for idx, job_state in self._my_partial._job_states.items()
        }

    def get_job_status_for_all_reals(
        self,
    ) -> Mapping[Tuple[str, str], Union[str, datetime.datetime]]:
        return {
            idx: job_state["status"]
            for idx, job_state in self._my_partial._job_states.items()
        }

    @property
    def reals(self) -> Mapping[str, "RealizationSnapshot"]:
        return self._my_partial.reals

    def get_jobs_for_real(self, real_id: str) -> Dict[str, "Job"]:
        return {
            job_idx[1]: Job(**job_data)
            for job_idx, job_data in self._my_partial._job_states.items()
            if job_idx[0] == real_id
        }

    def get_real(self, real_id: str) -> "RealizationSnapshot":
        return RealizationSnapshot(**self._my_partial._realization_states[real_id])

    def get_job(self, real_id: str, job_id: str) -> "Job":
        return Job(**self._my_partial._job_states[(real_id, job_id)])

    def get_successful_realizations(self) -> typing.List[int]:
        return [
            int(real_idx)
            for real_idx, real_data in self._my_partial._realization_states.items()
            if real_data[ids.STATUS] == state.REALIZATION_STATE_FINISHED
        ]

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


class RealizationSnapshot(BaseModel):
    status: Optional[str]
    active: Optional[bool]
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    jobs: Dict[str, Job] = {}


class SnapshotDict(BaseModel):
    status: Optional[str] = state.ENSEMBLE_STATE_UNKNOWN
    reals: Dict[str, RealizationSnapshot] = {}
    metadata: Dict[str, Any] = {}


class SnapshotBuilder(BaseModel):
    jobs: Dict[str, Job] = {}
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
                jobs=self.jobs,
                start_time=start_time,
                end_time=end_time,
                status=status,
            )
        return Snapshot(top.dict())

    def add_job(
        self,
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
        self.jobs[job_id] = Job(
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
        for job_id, job in realization_data.get("jobs", {}).items():
            job_idx = (real_id, job_id)
            partial._job_states[job_idx] = job

    return partial
