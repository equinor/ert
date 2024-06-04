import datetime
import re
import typing
from collections import defaultdict
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

from cloudevents.http import CloudEvent
from dateutil.parser import parse
from pydantic import BaseModel, ConfigDict
from qtpy.QtGui import QColor

from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator import state

_regexp_pattern = r"(?<=/{token}/)[^/]+"


def _match_token(token: str, source: str) -> str:
    f_pattern = _regexp_pattern.format(token=token)
    match = re.search(f_pattern, source)
    return match if match is None else match.group()  # type: ignore


def _get_real_id(source: str) -> str:
    return _match_token("real", source)


def _get_forward_model_id(source: str) -> str:
    return _match_token("forward_model", source)


def _get_forward_model_index(source: str) -> str:
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
    ids.EVTYPE_FORWARD_MODEL_START: state.FORWARD_MODEL_STATE_START,
    ids.EVTYPE_FORWARD_MODEL_RUNNING: state.FORWARD_MODEL_STATE_RUNNING,
    ids.EVTYPE_FORWARD_MODEL_SUCCESS: state.FORWARD_MODEL_STATE_FINISHED,
    ids.EVTYPE_FORWARD_MODEL_FAILURE: state.FORWARD_MODEL_STATE_FAILURE,
}

_ENSEMBLE_TYPE_EVENT_TO_STATUS = {
    ids.EVTYPE_ENSEMBLE_STARTED: state.ENSEMBLE_STATE_STARTED,
    ids.EVTYPE_ENSEMBLE_STOPPED: state.ENSEMBLE_STATE_STOPPED,
    ids.EVTYPE_ENSEMBLE_CANCELLED: state.ENSEMBLE_STATE_CANCELLED,
    ids.EVTYPE_ENSEMBLE_FAILED: state.ENSEMBLE_STATE_FAILED,
}


def convert_iso8601_to_datetime(
    timestamp: Union[datetime.datetime, str],
) -> datetime.datetime:
    if isinstance(timestamp, datetime.datetime):
        return timestamp

    return parse(timestamp)


RealId = str
FmStepId = str


class SnapshotMetadata(TypedDict, total=False):
    aggr_job_status_colors: DefaultDict[RealId, Dict[FmStepId, QColor]]
    real_status_colors: Dict[RealId, QColor]
    sorted_real_ids: List[RealId]
    sorted_forward_model_ids: DefaultDict[RealId, List[FmStepId]]


def _filter_nones(some_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in some_dict.items() if value is not None}


class Snapshot:
    def __init__(self) -> None:
        self._realization_states: Dict[
            str, Dict[str, Union[bool, datetime.datetime, str]]
        ] = defaultdict(dict)
        """A shallow dictionary of realization states. The key is a string with
        realization number, pointing to a dict with keys active (bool),
        start_time (datetime), end_time (datetime) and status (str)."""

        self._forward_model_states: Dict[
            Tuple[str, str], Dict[str, Union[str, datetime.datetime]]
        ] = defaultdict(dict)
        """A shallow dictionary of forward_model states. The key is a tuple of two
        strings with realization id and forward_model id, pointing to a dict with
        the same members as the ForwardModel."""

        self._ensemble_state: Optional[str] = None
        # TODO not sure about possible values at this point, as GUI hijacks this one as
        # well
        self._metadata = SnapshotMetadata(
            aggr_job_status_colors=defaultdict(dict),
            real_status_colors=dict(),
            sorted_real_ids=list(),
            sorted_forward_model_ids=defaultdict(list),
        )

    def update_realization(
        self,
        real_id: str,
        status: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> "Snapshot":
        self._realization_states[real_id].update(
            _filter_nones(
                {"status": status, "start_time": start_time, "end_time": end_time}
            )
        )
        return self

    def update_forward_model(
        self,
        real_id: str,
        forward_model_id: str,
        forward_model: "ForwardModel",
    ) -> "Snapshot":
        forward_model_update = _filter_nones(forward_model.model_dump())
        self._forward_model_states[(real_id, forward_model_id)].update(
            forward_model_update
        )
        return self

    def get_forward_model_status_for_all_reals(
        self,
    ) -> Mapping[Tuple[str, str], Union[str, datetime.datetime]]:
        return {
            idx: forward_model_state["status"]
            for idx, forward_model_state in self._forward_model_states.items()
        }

    @property
    def reals(self) -> Dict[str, "RealizationSnapshot"]:
        return {
            real_id: RealizationSnapshot(**real_data)
            for real_id, real_data in self._realization_states.items()
        }

    def get_real_ids(self) -> Sequence[str]:
        """we can have information about realizations in both _realization_states and
        _forward_model_states - we combine the existing IDs"""
        real_ids = []
        for idx in self._forward_model_states:
            real_id = idx[0]
            if real_id not in real_ids:
                real_ids.append(real_id)
        for real_id in self._realization_states:
            if real_id not in real_ids:
                real_ids.append(real_id)
        return sorted(real_ids, key=int)

    def to_dict(self) -> Dict[str, Any]:
        """Used to send snapshot updates"""
        _dict: Dict[str, Any] = {}
        if self._metadata:
            _dict["metadata"] = self._metadata
        if self._ensemble_state:
            _dict["status"] = self._ensemble_state
        if self._realization_states:
            _dict["reals"] = self._realization_states

        for fm_tuple, fm_values_dict in self._forward_model_states.items():
            real_id = fm_tuple[0]
            if "reals" not in _dict:
                _dict["reals"] = {}
            if real_id not in _dict["reals"]:
                _dict["reals"][real_id] = {}
            if "forward_models" not in _dict["reals"][real_id]:
                _dict["reals"][real_id]["forward_models"] = {}

            forward_model_id = fm_tuple[1]
            _dict["reals"][real_id]["forward_models"][forward_model_id] = fm_values_dict

        return _dict

    def _merge(self, other: "Snapshot") -> "Snapshot":
        self._metadata.update(other._metadata)
        if other._ensemble_state is not None:
            self._ensemble_state = other._ensemble_state
        for real_id, other_real_data in other._realization_states.items():
            self._realization_states[real_id].update(other_real_data)
        for forward_model_id, other_fm_data in other._forward_model_states.items():
            self._forward_model_states[forward_model_id].update(other_fm_data)
        return self

    def update_from_cloudevent_and_generate_update_snapshot(
        self,
        event: CloudEvent,
        update_event_snapshot: Optional["Snapshot"] = None,
    ) -> "Snapshot":
        """Add the data from the cloudevent, and also generate a new snapshot containing only the updated data.

        Args:
            event (CloudEvent): The CloudEvent with the data to update.
            update_event_snapshot (Optional[&quot;Snapshot&quot;], optional): The snapshot which should only have the updated data attached. Returns a new snapshot if None is provided.

        Returns:
            Snapshot: A snapshot only containing the updated data.
        """
        if update_event_snapshot is None:
            update_event_snapshot = Snapshot()
        e_type = event["type"]
        if e_type in ids.EVGROUP_REALIZATION:
            update_event_snapshot = self._handle_realization_event(
                event, update_event_snapshot
            )

        elif e_type in ids.EVGROUP_FORWARD_MODEL:
            update_event_snapshot = self._handle_forward_model_event(
                event, update_event_snapshot
            )

        elif e_type in ids.EVGROUP_ENSEMBLE:
            self._ensemble_state = _ENSEMBLE_TYPE_EVENT_TO_STATUS[e_type]
            update_event_snapshot._ensemble_state = self._ensemble_state
        elif e_type == ids.EVTYPE_EE_SNAPSHOT_UPDATE:
            other_snapshot = Snapshot.from_nested_dict(event.data)
            self._merge(other_snapshot)
            update_event_snapshot._merge(other_snapshot)
        else:
            raise ValueError(f"Unknown type: {e_type}")
        return update_event_snapshot

    def _handle_realization_event(
        self, event: CloudEvent, update_event_snapshot: "Snapshot"
    ) -> "Snapshot":
        e_source = event["source"]
        e_type = event["type"]
        timestamp = event["time"]
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
        self.update_realization(_get_real_id(e_source), status, start_time, end_time)

        if e_type == ids.EVTYPE_REALIZATION_TIMEOUT:
            for (
                forward_model_id,
                forward_model,
            ) in self.get_forward_models_for_real(_get_real_id(e_source)).items():
                if forward_model.status != state.FORWARD_MODEL_STATE_FINISHED:
                    real_id = _get_real_id(e_source)
                    forward_model_idx = (real_id, forward_model_id)
                    if forward_model_idx not in self._forward_model_states:
                        self._forward_model_states[forward_model_idx] = {}
                    self._forward_model_states[forward_model_idx].update(
                        {
                            "status": state.FORWARD_MODEL_STATE_FAILURE,
                            "end_time": end_time,  # type: ignore
                            "error": "The run is cancelled due to "
                            "reaching MAX_RUNTIME",
                        }
                    )
        update_event_snapshot._realization_states = self._realization_states
        return update_event_snapshot

    def _handle_forward_model_event(
        self, event: CloudEvent, update_event_snapshot: "Snapshot"
    ) -> "Snapshot":
        e_source = event["source"]
        e_type = event["type"]
        timestamp = event["time"]
        status = _FM_TYPE_EVENT_TO_STATUS[e_type]
        start_time = None
        end_time = None
        error = None
        if e_type == ids.EVTYPE_FORWARD_MODEL_START:
            start_time = convert_iso8601_to_datetime(timestamp)
        elif e_type in {
            ids.EVTYPE_FORWARD_MODEL_SUCCESS,
            ids.EVTYPE_FORWARD_MODEL_FAILURE,
        }:
            end_time = convert_iso8601_to_datetime(timestamp)
            # Make sure error msg from previous failed run is replaced
            error = ""
            if event.data is not None:
                error = event.data.get(ids.ERROR_MSG)

        fm_dict = {
            ids.STATUS: status,
            ids.START_TIME: start_time,
            ids.END_TIME: end_time,
            ids.INDEX: _get_forward_model_index(e_source),
            ids.ERROR: error,
        }
        if e_type == ids.EVTYPE_FORWARD_MODEL_RUNNING:
            fm_dict[ids.CURRENT_MEMORY_USAGE] = event.data.get(ids.CURRENT_MEMORY_USAGE)
            fm_dict[ids.MAX_MEMORY_USAGE] = event.data.get(ids.MAX_MEMORY_USAGE)
        if e_type == ids.EVTYPE_FORWARD_MODEL_START:
            fm_dict[ids.STDOUT] = event.data.get(ids.STDOUT)
            fm_dict[ids.STDERR] = event.data.get(ids.STDERR)
        self.update_forward_model(
            _get_real_id(e_source),
            _get_forward_model_id(e_source),
            ForwardModel(**fm_dict),
        )
        update_event_snapshot._forward_model_states = self._forward_model_states
        return update_event_snapshot

    def merge_event(self, event: "Snapshot") -> None:
        self._merge(event)

    def merge(self, update_as_nested_dict: Mapping[str, Any]) -> None:
        self._merge(Snapshot.from_nested_dict(update_as_nested_dict))

    def merge_metadata(self, metadata: SnapshotMetadata) -> None:
        """only used in gui snapshot model, which only cares about the
        snapshot's metadata"""
        self._metadata.update(metadata)

    @property
    def status(self) -> Optional[str]:
        return self._ensemble_state

    @property
    def metadata(self) -> SnapshotMetadata:
        return self._metadata

    def get_all_forward_models(
        self,
    ) -> Mapping[Tuple[str, str], "ForwardModel"]:
        return {
            idx: ForwardModel(**forward_model_state)
            for idx, forward_model_state in self._forward_model_states.items()
        }

    def get_forward_models_for_real(self, real_id: str) -> Dict[str, "ForwardModel"]:
        return {
            forward_model_idx[1]: ForwardModel(**forward_model_data)
            for forward_model_idx, forward_model_data in self._forward_model_states.items()
            if forward_model_idx[0] == real_id
        }

    def get_real(self, real_id: str) -> "RealizationSnapshot":
        return RealizationSnapshot(**self._realization_states[real_id])

    def get_job(self, real_id: str, forward_model_id: str) -> "ForwardModel":
        return ForwardModel(**self._forward_model_states[(real_id, forward_model_id)])

    def get_successful_realizations(self) -> typing.List[int]:
        return [
            int(real_idx)
            for real_idx, real_data in self._realization_states.items()
            if real_data[ids.STATUS] == state.REALIZATION_STATE_FINISHED
        ]

    def aggregate_real_states(self) -> typing.Dict[str, int]:
        states: Dict[str, int] = defaultdict(int)
        for real in self._realization_states.values():
            status = real["status"]
            assert isinstance(status, str)
            states[status] += 1
        return states

    def data(self) -> Mapping[str, Any]:
        # The gui uses this
        return self.to_dict()

    @classmethod
    def from_nested_dict(cls, data: Mapping[str, Any]) -> "Snapshot":
        snapshot = Snapshot()
        if "metadata" in data:
            snapshot._metadata = data["metadata"]
        if "status" in data:
            snapshot._ensemble_state = data["status"]
        for real_id, realization_data in data.get("reals", {}).items():
            snapshot._realization_states[real_id] = _filter_nones(
                {
                    "status": realization_data.get("status"),
                    "active": realization_data.get("active"),
                    "start_time": realization_data.get("start_time"),
                    "end_time": realization_data.get("end_time"),
                }
            )
            for forward_model_id, job in realization_data.get(
                "forward_models", {}
            ).items():
                forward_model_idx = (real_id, forward_model_id)
                snapshot._forward_model_states[forward_model_idx] = job

        return snapshot


class ForwardModel(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    status: Optional[str]
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    index: Optional[str] = None
    current_memory_usage: Optional[str] = None
    max_memory_usage: Optional[str] = None
    name: Optional[str] = None
    error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class RealizationSnapshot(BaseModel):
    status: Optional[str] = None
    active: Optional[bool] = None
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    forward_models: Dict[str, ForwardModel] = {}


class SnapshotDict(BaseModel):
    status: Optional[str] = state.ENSEMBLE_STATE_UNKNOWN
    reals: Dict[str, RealizationSnapshot] = {}
    metadata: Dict[str, Any] = {}


class SnapshotBuilder(BaseModel):
    forward_models: Dict[str, ForwardModel] = {}
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
                forward_models=self.forward_models,
                start_time=start_time,
                end_time=end_time,
                status=status,
            )
        return Snapshot.from_nested_dict(top.model_dump())

    def add_forward_model(
        self,
        forward_model_id: str,
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
        self.forward_models[forward_model_id] = ForwardModel(
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
