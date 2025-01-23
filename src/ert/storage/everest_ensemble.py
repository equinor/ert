from .local_ensemble import LocalEnsemble


class EverestEnsemble:
    def __init__(self, ert_ensemble: LocalEnsemble):
        self._ert_ensemble = ert_ensemble

    @property
    def ert_ensemble(self) -> LocalEnsemble:
        return self._ert_ensemble
