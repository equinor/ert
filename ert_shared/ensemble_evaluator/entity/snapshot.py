from ert_shared.ensemble_evaluator.entity.tool import recursive_update


class Snapshot:

    def __init__(self, dict):
        self._data = dict

    def merge_event(self, event):
        recursive_update(self._data, event)

    def to_dict(self):
        return self._data
