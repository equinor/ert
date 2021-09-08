"""Module for tracking the states of a tree of state machines.
"""
from collections import deque
import io
import json
import asyncio
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Type, Union
import ert_shared.status.entity.state as state
import ert_shared.ensemble_evaluator.entity.identifiers as ids
from json import JSONEncoder
from copy import deepcopy
from cloudevents.http import from_json, to_json
from cloudevents.http.event import CloudEvent
from dateutil.parser import parse
import datetime


def convert_iso8601_to_datetime(timestamp):
    if isinstance(timestamp, datetime.datetime):
        return timestamp

    return parse(timestamp)


class _Encoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class _State:
    def __init__(self, name: str, data: Optional[dict] = None) -> None:
        self.name = name
        self.data = data

    def __repr__(self) -> str:
        repr_s = f"State<{self.name}"
        if self.data:
            repr_s += f" with data {self.data}"
        repr_s += ">"
        return repr_s

    def __eq__(self, other: object):
        if isinstance(other, _State):
            return self.name == other.name and (
                # if either has data, compare it
                (not other.data and not self.data)
                or other.data == self.data
            )
        return self == other

    def with_data(self, data: Optional[dict]) -> "_State":
        """Add data to a copy of this state."""
        if data is None:
            return self

        s_copy = deepcopy(self)
        s_copy.data = data
        return s_copy


_ENSEMBLE_STARTED = _State(state.ENSEMBLE_STATE_STARTED)
_ENSEMBLE_STOPPED = _State(state.ENSEMBLE_STATE_STOPPED)
_ENSEMBLE_CANCELLED = _State(state.ENSEMBLE_STATE_CANCELLED)
_ENSEMBLE_FAILED = _State(state.ENSEMBLE_STATE_FAILED)
_ENSEMBLE_UNKNOWN = _State(state.ENSEMBLE_STATE_UNKNOWN)

_REALIZATION_WAITING = _State(state.REALIZATION_STATE_WAITING)
_REALIZATION_PENDING = _State(state.REALIZATION_STATE_PENDING)
_REALIZATION_RUNNING = _State(state.REALIZATION_STATE_RUNNING)
_REALIZATION_FAILED = _State(state.REALIZATION_STATE_FAILED)
_REALIZATION_FINISHED = _State(state.REALIZATION_STATE_FINISHED)
_REALIZATION_UNKNOWN = _State(state.REALIZATION_STATE_UNKNOWN)

_STEP_WAITING = _State(state.STEP_STATE_WAITING)
_STEP_PENDING = _State(state.STEP_STATE_PENDING)
_STEP_RUNNING = _State(state.STEP_STATE_RUNNING)
_STEP_FAILURE = _State(state.STEP_STATE_FAILURE)
_STEP_SUCCESS = _State(state.STEP_STATE_SUCCESS)
_STEP_UNKNOWN = _State(state.STEP_STATE_UNKNOWN)

_JOB_START = _State(state.JOB_STATE_START)
_JOB_RUNNING = _State(state.JOB_STATE_RUNNING)
_JOB_FINISHED = _State(state.JOB_STATE_FINISHED)
_JOB_FAILURE = _State(state.JOB_STATE_FAILURE)

_FSM_UNKNOWN = _State("__unknown__")


class _Transition(NamedTuple):
    from_state: _State
    to_state: _State

    def comparable_to(self, from_state: _State, to_state: _State) -> bool:
        """Disregarding node, are the two transitions roughly comparable?"""
        return (
            self.from_state.name == from_state.name
            and self.to_state.name == to_state.name
        )

    def __repr__(self) -> str:
        return f"Transition<{self.from_state} -> {self.to_state}>"


class _StateChange(NamedTuple):
    transition: _Transition
    node: "_FSM"

    def __repr__(self) -> str:
        return f"StateChange<{self.transition}@{self.node}>"

    def __lt__(self, other) -> bool:
        return self.node.src < other.node.src

    def __getitem__(self, attr: Any) -> Any:
        if attr == "source":
            return self.node.src
        # We are not using super() here due to https://bugs.python.org/issue41629
        return tuple.__getitem__(self, attr)

    @property
    def src(self) -> str:
        return self["source"]

    @staticmethod
    def changeset_to_partial(changes: List["_StateChange"]) -> dict:
        """Create a partial from a list of state changes."""
        partial: Dict[str, Any] = {}

        # Each key represents an entity which is mapped to from iterating over
        # the path of the change. So given /foo/bar/baz/qux, foo is None, qux
        # maps to "jobs".
        partial_key_hierarchy: List[str] = [None, ids.REALS, ids.STEPS, ids.JOBS]

        sub_partial = partial
        for change in changes:
            parts = change.node.src.split("/")[1:]

            # Iteratively build a partial dict while looping over the parts of
            # the path e.g. /foo/bar/baz/qux. Given that path, qux is the
            # terminal part, and foo represents the "reals", qux the "jobs",
            # etc.
            # TODO: handle multiple changes for e.g. one job
            for pos, part in enumerate(parts):
                is_terminal_part = pos == len(parts) - 1
                key = partial_key_hierarchy[pos]

                if key and key not in sub_partial:
                    sub_partial[key] = {part: {}}
                    sub_partial = sub_partial[key][part]
                elif key in sub_partial:
                    if not part in sub_partial[key]:
                        sub_partial[key][part] = {}
                    sub_partial = sub_partial[key][part]

                if is_terminal_part:
                    sub_partial[ids.STATUS] = change.transition.to_state.name
                    if change.transition.to_state.data:
                        sub_partial[ids.DATA] = change.transition.to_state.data
                    sub_partial = partial

        return partial


class IllegalTransition(Exception):
    """Represents an illegal transition."""

    def __init__(self, error: str, node: "_FSM", transition: _Transition) -> None:
        super().__init__(error)
        self.node = node
        self.transition = transition

    def __getitem__(self, attr: Any) -> Any:
        if attr == "source":
            return self.node.src
        return super().__getitem__(attr)

    @property
    def src(self) -> str:
        return self["source"]


_TransitionResult = Union[_StateChange, IllegalTransition]
_TransitionTrigger = Union[CloudEvent, _StateChange]


class _FSM:
    def __init__(self, branch: str, id_: str, parent: Optional["_FSM"] = None) -> None:
        self._id = id_
        self._branch = branch
        self._parent: Optional[_FSM] = parent
        self._children: List[_FSM] = []
        if self._parent:
            self._parent._children.append(self)

        self._state: _State = _FSM_UNKNOWN
        self._transitions: List[_Transition] = []

    def __repr__(self) -> str:
        return f"Node<{self.__class__.__name__}@{self.src}>"

    @property
    def id_(self) -> str:
        """The id property."""
        return self._id

    @property
    def children(self) -> List["_FSM"]:
        """The children property."""
        return self._children

    @property
    def src(self) -> str:
        """Return the source of this node, which is a textual presentation of
        this nodes position in the tree in the form of /foo/bar/baz.
        """
        return self._branch + self._id

    @property
    def path(self) -> str:
        """Return the path of this node, which is the src property plus a
        trailing slash.
        """
        return self.src + "/"

    @property
    def state(self) -> _State:
        """The state property"""
        return self._state

    @state.setter
    def state(self, state: _State) -> None:
        """Set the state of this node."""
        prev_data = self._state.data
        self._state = state.with_data(prev_data)

    def transition(self, to_state: _State) -> Generator[_TransitionResult, None, None]:
        """Transition this node to a new state."""
        for trans in self._transitions:
            if trans.comparable_to(self.state, to_state):

                # these states are equal even considering data, thus no-op
                if trans.from_state == to_state:
                    break

                self.state = to_state
                yield _StateChange(
                    transition=_Transition(trans.from_state, to_state), node=self
                )
                break
        else:
            yield IllegalTransition(
                f"no transition for {self} from {self.state} -> {to_state}",
                self,
                _Transition(self.state, to_state),
            )

    def add_transition(self, transition: _Transition) -> None:
        """Add a transition."""
        self._transitions.append(transition)

    def is_applicable(self, obj: _TransitionTrigger) -> bool:
        """Return whether or not the event is applicable to this node."""
        return obj["source"].startswith(self.src)

    def _dispatch_event(
        self, event: CloudEvent
    ) -> Generator[_TransitionResult, None, None]:
        if self.children:
            for child in self.children:
                yield from child.dispatch(event)
        else:
            yield from ()

    def _dispatch_state_change(
        self, state_change: _StateChange
    ) -> Generator[_TransitionResult, None, None]:
        if self.children:
            for child in self.children:
                yield from child.dispatch(state_change)
        else:
            yield from ()

    def _dispatch_illegal_transition(
        self, illegal_transition: IllegalTransition
    ) -> Generator[_TransitionResult, None, None]:
        """Should not be called at all if the illegal_transition has been
        handled."""
        if self.children:
            for child in self.children:
                yield from child.dispatch(illegal_transition)
        else:
            raise illegal_transition

    def dispatch(
        self, obj: _TransitionTrigger
    ) -> Generator[_TransitionResult, None, None]:
        """Dispatch something that triggers changes."""
        if not self.is_applicable(obj):
            return
        if isinstance(obj, CloudEvent):
            yield from self._dispatch_event(obj)
        elif isinstance(obj, _StateChange):
            yield from self._dispatch_state_change(obj)
        elif isinstance(obj, IllegalTransition):
            yield from self._dispatch_illegal_transition(obj)
        else:
            raise TypeError(f"cannot dispatch {type(obj)}")


class JobFSM(_FSM):
    def __init__(self, id_: str, parent: _FSM) -> None:
        super().__init__(parent.path, id_, parent=parent)
        self._state: _State = _JOB_START
        self.add_transition(_Transition(_JOB_START, _JOB_RUNNING))
        self.add_transition(_Transition(_JOB_RUNNING, _JOB_RUNNING))
        self.add_transition(_Transition(_JOB_RUNNING, _JOB_FINISHED))
        self.add_transition(_Transition(_JOB_START, _JOB_FAILURE))
        self.add_transition(_Transition(_JOB_START, _JOB_FINISHED))
        self.add_transition(_Transition(_JOB_RUNNING, _JOB_FAILURE))

    def _dispatch_event(
        self, event: CloudEvent
    ) -> Generator[_TransitionResult, None, None]:
        data = event.data if event.data else {}
        timestamp = convert_iso8601_to_datetime(event["time"])

        if event[ids.TYPE] == ids.EVTYPE_FM_JOB_START:
            data[ids.START_TIME] = timestamp
        elif event[ids.TYPE] in (ids.EVTYPE_FM_JOB_SUCCESS, ids.EVTYPE_FM_JOB_FAILURE):
            data[ids.END_TIME] = timestamp

        if event[ids.TYPE] == ids.EVTYPE_FM_JOB_START:
            yield from self.transition(_JOB_RUNNING.with_data(data))
        elif event[ids.TYPE] == ids.EVTYPE_FM_JOB_RUNNING:
            yield from self.transition(_JOB_RUNNING.with_data(data))
        elif event[ids.TYPE] == ids.EVTYPE_FM_JOB_SUCCESS:
            yield from self.transition(_JOB_FINISHED.with_data(data))
        elif event[ids.TYPE] == ids.EVTYPE_FM_JOB_FAILURE:
            yield from self.transition(_JOB_FAILURE.with_data(data))

        yield from super()._dispatch_event(event)


class StepFSM(_FSM):
    def __init__(self, id_: str, parent: "_FSM") -> None:
        super().__init__(parent.path, id_, parent=parent)
        self._state = _STEP_UNKNOWN
        self.add_transition(_Transition(_STEP_UNKNOWN, _STEP_RUNNING))
        self.add_transition(_Transition(_STEP_RUNNING, _STEP_RUNNING))
        self.add_transition(_Transition(_STEP_RUNNING, _STEP_SUCCESS))

    def _dispatch_event(
        self, event: CloudEvent
    ) -> Generator[_TransitionResult, None, None]:
        data = event.data if event.data else {}
        timestamp = convert_iso8601_to_datetime(event["time"])
        if event[ids.TYPE] == ids.EVTYPE_FM_STEP_RUNNING:
            data[ids.START_TIME] = timestamp
        elif event[ids.TYPE] in (
            ids.EVTYPE_FM_STEP_FAILURE,
            ids.EVTYPE_FM_STEP_SUCCESS,
            ids.EVTYPE_FM_STEP_TIMEOUT,
        ):
            data[ids.END_TIME] = timestamp

        if event[ids.TYPE] == ids.EVTYPE_FM_STEP_RUNNING:
            yield from self.transition(_STEP_RUNNING.with_data(data))
        elif event[ids.TYPE] == ids.EVTYPE_FM_STEP_SUCCESS:
            yield from self.transition(_STEP_SUCCESS.with_data(data))
        elif event[ids.TYPE] == ids.EVTYPE_FM_STEP_TIMEOUT:
            for child in self.children:
                if child.state.name != _JOB_FINISHED.name:
                    job_error = "The run is cancelled due to reaching MAX_RUNTIME"
                    yield from child.transition(
                        _JOB_FAILURE.with_data(
                            {ids.ERROR: job_error, ids.ids.END_TIME: timestamp}
                        )
                    )

        yield from super()._dispatch_event(event)

    def _dispatch_illegal_transition(
        self, illegal_transition: IllegalTransition
    ) -> Generator[_TransitionResult, None, None]:
        if illegal_transition.node in self.children:
            if illegal_transition.transition.to_state == _JOB_FINISHED:
                # Assume it, and the jobs preceding it, are succeeding
                i = self._children.index(illegal_transition.node)
                while i >= 0:
                    old_state = self._children[i].state
                    self._children[i].state = _JOB_FINISHED
                    yield _StateChange(
                        transition=_Transition(old_state, self._children[i].state),
                        node=self._children[i],
                    )
                    i -= 1

                # since it was handled here, stop propagating it downwards
                return

        yield from super()._dispatch_illegal_transition(illegal_transition)


class RealizationFSM(_FSM):
    def __init__(self, id_: str, parent: _FSM) -> None:
        super().__init__(parent.path, id_, parent=parent)
        self._state = _REALIZATION_UNKNOWN
        self.add_transition(_Transition(_REALIZATION_UNKNOWN, _REALIZATION_RUNNING))
        self.add_transition(_Transition(_REALIZATION_RUNNING, _REALIZATION_RUNNING))
        self.add_transition(_Transition(_REALIZATION_RUNNING, _REALIZATION_FINISHED))

    def _dispatch_state_change(
        self, state_change: _StateChange
    ) -> Generator[_TransitionResult, None, None]:
        if state_change.node in self.children:
            if state_change.transition.to_state == _STEP_RUNNING:
                yield from self.transition(_REALIZATION_RUNNING)
            elif state_change.transition.to_state == _STEP_SUCCESS:
                # are all steps succeeding?
                for step in self._children:
                    if step.state != _STEP_SUCCESS:
                        break
                else:
                    yield from self.transition(_REALIZATION_FINISHED)

        yield from super()._dispatch_state_change(state_change)


class EnsembleFSM(_FSM):
    def __init__(self, id_: str) -> None:
        super().__init__("/", id_, parent=None)
        self._state = _ENSEMBLE_UNKNOWN
        self.metadata: Dict[Any, Any] = {}

        self.add_transition(_Transition(_ENSEMBLE_UNKNOWN, _ENSEMBLE_STARTED))
        self.add_transition(_Transition(_ENSEMBLE_STARTED, _ENSEMBLE_STARTED))
        self.add_transition(_Transition(_ENSEMBLE_STARTED, _ENSEMBLE_STOPPED))
        self.add_transition(_Transition(_ENSEMBLE_STOPPED, _ENSEMBLE_STOPPED))

    def snapshot_dict(self) -> dict:
        """Return a snapshot representing the state of this and all other
        descendant nodes."""
        snapshot: Dict[str, Any] = {ids.STATUS: self.state.name, ids.REALS: {}}

        if self.metadata:
            snapshot[ids.METADATA] = self.metadata

        for real in self.children:
            snapshot[ids.REALS][real.id_] = {ids.STATUS: real.state.name, ids.STEPS: {}}
            if real.state.data:
                snapshot[ids.REALS][real.id_][ids.DATA] = real.state.data

            for step in real.children:
                step_d: Dict[str, Any] = {
                    ids.STATUS: step.state.name,
                    ids.JOBS: {},
                }
                if step.state.data:
                    step_d[ids.DATA] = step.state.data
                snapshot[ids.REALS][real.id_][ids.STEPS][step.id_] = step_d
                for job in step.children:
                    job_d: Dict[str, Any] = {ids.STATUS: job.state.name}
                    if job.state.data:
                        job_d[ids.DATA] = job.state.data
                    snapshot[ids.REALS][real.id_][ids.STEPS][step.id_][ids.JOBS][
                        job.id_
                    ] = job_d
        return snapshot

    def _dispatch_event(
        self, event: CloudEvent
    ) -> Generator[_TransitionResult, None, None]:
        if event[ids.TYPE] == ids.EVTYPE_ENSEMBLE_STARTED:
            yield from self.transition(_ENSEMBLE_STARTED.with_data(event.data))

        if event[ids.TYPE] == ids.EVTYPE_ENSEMBLE_STOPPED:
            yield from self.transition(_ENSEMBLE_STOPPED.with_data(event.data))

        yield from super()._dispatch_event(event)

    def _recursive_dispatch(
        self, change: _TransitionTrigger
    ) -> Generator[_TransitionResult, None, None]:
        deck = deque([change])
        max_iterations = 1000
        iterations = 0
        while len(deck):
            iterations += 1
            trigger = deck.pop()
            changes = list(super().dispatch(trigger))
            deck.extendleft(changes)
            yield from changes

            if iterations > max_iterations:
                raise RecursionError(f"{change} caused > {max_iterations} changes")

    def dispatch(
        self, obj: _TransitionTrigger
    ) -> Generator[_TransitionResult, None, None]:
        """Dispatch something that triggers a change, but recurse over the tree
        for all changes until there are no more."""
        if not self.is_applicable(obj):
            return
        if not isinstance(obj, CloudEvent) and not isinstance(obj, IllegalTransition):
            raise TypeError(f"cannot dispatch {type(obj)}")
        for change in self._dispatch_event(obj):
            yield change
            yield from self._recursive_dispatch(change)

    def dispatch_cloud_events_and_return_partial(
        self, events: CloudEvent
    ) -> Dict[Any, Any]:
        changes: List[_StateChange] = []
        for event in events:
            for change in self.dispatch(event):
                if not isinstance(change, _StateChange):
                    continue
                changes.append(change)
        return _StateChange.changeset_to_partial(changes)

    @staticmethod
    def from_ert_trivial_graph_format(tgf: str) -> "EnsembleFSM":
        """Takes ERT Trivial Graph Format (ERTGRAF) and converts it into an EnsembleFSM
        FSM. The ERTGRAF is defined to be directed, acyclical and formatted as follows:
            <type of node> <source>
            …
            <type of node> <source>

        Where the type of node can be one of Ensemble, Realization, Step and
        Job. Source is the path of the node, e.g. /foo/bar/baz/qux. A trivial
        example is
            Ensemble /0
            Realization /0/0
            Realization /0/1
            Step /0/0/0
            Job /0/0/0/0
        """
        type_map: Dict[str, Any] = {
            "Ensemble": EnsembleFSM,
            "Realization": RealizationFSM,
            "Step": StepFSM,
            "Job": JobFSM,
        }
        source_to_type: Dict[str, Any] = {}
        s = io.StringIO(tgf)
        for line in s.readlines():
            line = line.strip()
            if not line:
                continue
            wanted_type, source = line.split(" ")
            if wanted_type not in type_map:
                raise TypeError(f"unexpected type {wanted_type}")
            source_to_type[source] = wanted_type

        source_to_instance: Dict[str, _FSM] = {}
        root = None
        for source in sorted(
            source_to_type.keys(), key=lambda source: len(source.split("/"))
        ):
            parts = source.split("/")[1:]
            if len(parts) - 1 == 0:  # root
                source_to_instance[source] = type_map[source_to_type[source]](
                    parts[len(parts) - 1]
                )
                root = source_to_instance[source]
            else:
                parent = source_to_instance["/" + "/".join(parts[:-1])]
                source_to_instance[source] = type_map[source_to_type[source]](
                    parts[len(parts) - 1], parent
                )
        return root
