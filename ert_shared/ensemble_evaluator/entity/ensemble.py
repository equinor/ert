from ert_shared.ensemble_evaluator.entity.snapshot import Snapshot


class _Ensemble:
    def __init__(self, data):
        self._data = Snapshot(data)

    def evaluate(self, host, port):
        pass

    def snapshot(self):
        return self._data

    def to_dict(self):
        return self._data
