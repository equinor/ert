class _Ensemble:
    def __init__(self, data):
        self._data = data

    def snapshot(self):
        return self.to_dict()

    def to_dict(self):
        return self._data
