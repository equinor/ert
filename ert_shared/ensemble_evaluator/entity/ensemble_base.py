class _Ensemble:
    def __init__(self, reals, metadata):
        self._reals = reals
        self._metadata = metadata

    def __repr__(self):
        return f"Ensemble with {len(self._reals)} members"

    def evaluate(self, host, port):
        pass

    def get_reals(self):
        return self._reals

    def get_metadata(self):
        return self._metadata
