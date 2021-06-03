from ert_shared.ensemble_evaluator.entity import identifiers as ids


class _Command:
    def __init__(self, action):
        self._action = action

    def is_terminate(self):
        return self._action == ids.TERMINATE

    def is_pause(self):
        return self._action == ids.PAUSE

    @classmethod
    def from_dict(cls, data):
        return cls(data[ids.ACTION])

    def to_dict(self):
        return {ids.ACTION: self._action}

    def __eq__(self, other):
        return self._action == other._action


def create_command_pause():
    return _Command(ids.PAUSE)


def create_command_terminate():
    return _Command(ids.TERMINATE)


def create_command_from_dict(data):
    return _Command.from_dict(data)
