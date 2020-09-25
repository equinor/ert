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
        return self._forward_models is not None

    def to_dict(self):
        return {
            _STATUS: self._status,
            _FORWARD_MODELS: self._forward_models,
            _REALIZATIONS: self._realizations,
            _EVENT_INDEX: self._event_index,
        }

    @classmethod
    def from_dict(cls, data):
        event_index = data[_EVENT_INDEX]
        forward_models = data.get(_FORWARD_MODELS)
        realizations = data.get(_REALIZATIONS)
        status = data.get(_STATUS)
        return cls(
            event_index,
            forward_models=forward_models,
            realizations=realizations,
            status=status,
        )


def create_evaluator_event(event_index, forward_models, realizations, status):
    return _EnsembleResponseEvent(
        event_index,
        forward_models=forward_models,
        realizations=realizations,
        status=status,
    )


def create_evaluator_event_from_dict(data):
    return _EnsembleResponseEvent.from_dict(data)
