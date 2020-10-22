from ert_shared.ensemble_evaluator.entity.snapshot import Snapshot


class _Ensemble:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def evaluate(self, host, port):
        pass

    def forward_model_description(self):
        return self._snapshot

    def to_dict(self):
        return self._data.to_dict()
