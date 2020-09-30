import collections.abc
import copy
import json


def _recursive_update(d, u):
    for k, v in u.items():
        if k not in d:
            raise ValueError(f"Illegal field {k}")
        if isinstance(v, collections.abc.Mapping):
            d_val = d.get(k, {})
            if d_val is None:
                d[k] = copy.deepcopy(v)
            else:
                d[k] = _recursive_update(d_val, v)
        else:
            d[k] = v
    return d


_TERMINATE = "terminate"
_PAUSE = "pause"
_ACTION = "action"
_TERMINATED = "terminated"
_DONE = "done"
_STATUS = "status"
_FORWARD_MODELS = "forward_models"
_REALIZATIONS = "realizations"
_EVENT_INDEX = "event_index"


class _Command:
    def __init__(self, action):
        self._action = action

    def is_terminate(self):
        return self._action == _TERMINATE

    def is_pause(self):
        return self._action == _PAUSE

    @classmethod
    def from_dict(cls, data):
        return cls(data[_ACTION])

    def to_dict(self):
        return {_ACTION: self._action}


def create_command_pause():
    return _Command(_PAUSE)


def create_command_terminate():
    return _Command(_TERMINATE)


def create_command_from_dict(data):
    return _Command.from_dict(data)


_FMJ_ID = "id"
_FMJ_TYPE = "type"
_FMJ_INPUTS = "inputs"


class _ForwardModelJob:
    def __init__(self, fmj_id, fmj_type, inputs=()):
        self.id = fmj_id
        self.type = fmj_type
        self.inputs = inputs

    def to_dict(self):
        return {
            _FMJ_ID: self.id,
            _FMJ_TYPE: self.type,
            _FMJ_INPUTS: self.inputs,
        }

    @classmethod
    def from_dict(cls, data):
        fmj_id = data[_FMJ_ID]
        fmj_type = data[_FMJ_TYPE]
        inputs = data[_FMJ_INPUTS]
        return cls(
            fmj_id=fmj_id,
            fmj_type=fmj_type,
            inputs=inputs,
        )


def create_forward_model_job(fmj_id, fmj_type, inputs=()):
    return _ForwardModelJob(fmj_id=fmj_id, fmj_type=fmj_type, inputs=inputs)


_REALIZATION_STATUS = "status"
_REALIZATION_FORWARD_MODELS = "forward_models"


class _Realization:
    def __init__(self, status, forward_models):
        self._status = status
        self._forward_models = forward_models

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        return {
            _REALIZATION_STATUS: self._status,
            _REALIZATION_FORWARD_MODELS: self._forward_models,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data[_REALIZATION_STATUS],
            data[_REALIZATION_FORWARD_MODELS],
        )


def create_realization(status, forward_model):
    return _Realization(status, forward_model)


class RealizationDecoder(json.JSONEncoder):
    def default(self, o):
        return o.to_dict()


# TODO: might subclass EnsembleResponse?
class _EnsembleResponseEvent:
    def __init__(
        self, event_index, status=None, forward_models=None, realizations=None
    ):
        self._event_index = event_index
        self._status = status
        self._forward_models = forward_models
        self._realizations = realizations

    def __repr__(self):
        return str(self.to_dict())

    def is_terminated(self):
        return self._status == _TERMINATED

    def is_done(self):
        return self._status == _DONE

    def is_running(self):
        return not (self.is_terminated() or self.is_done())

    def is_partial(self):
        return self._forward_models is None

    def merge_event(self, other):
        if self.is_partial() or not other.is_partial():
            raise ValueError("Can only merge partial event into non-partial event")

        if self._event_index >= other._event_index:
            return

        self._status = self._status if other._status is None else other._status
        self._event_index = other._event_index

        _recursive_update(self._realizations, other._realizations)

    def to_dict(self):
        return {
            _STATUS: self._status,
            _FORWARD_MODELS: [fm.to_dict() for fm in self._forward_models]
            if self._forward_models is not None
            else None,
            _REALIZATIONS: self._realizations,
            _EVENT_INDEX: self._event_index,
        }

    @classmethod
    def from_dict(cls, data):
        event_index = data[_EVENT_INDEX]
        forward_models = data.get(_FORWARD_MODELS)
        if forward_models is not None:
            forward_models = [
                _ForwardModelJob.from_dict(fmj_dict) for fmj_dict in forward_models
            ]
        realizations = data.get(_REALIZATIONS)
        status = data.get(_STATUS)
        return cls(
            event_index,
            forward_models=forward_models,
            realizations=realizations,
            status=status,
        )


def create_evaluator_event(event_index, realizations, status):
    return _EnsembleResponseEvent(
        event_index,
        forward_models=None,
        realizations=realizations,
        status=status,
    )


def create_unindexed_evaluator_event(realizations, status):
    return create_evaluator_event(
        event_index=-1, realizations=realizations, status=status
    )


def create_evaluator_event_from_dict(data):
    return _EnsembleResponseEvent.from_dict(data)


def create_evaluator_snapshot(forward_models, realization_indexes):
    realizations = {}
    for realization_index in realization_indexes:
        realization_fmjs = {}
        for forward_model in forward_models:
            realization_fmjs[forward_model.id] = {"status": "unknown", "data": None}
        realizations[realization_index] = {
            "status": "unknown",
            "forward_models": realization_fmjs,
        }
    return _EnsembleResponseEvent(
        event_index=0,
        status="unknown",
        forward_models=forward_models,
        realizations=realizations,
    )
