from functools import cached_property

from ert.storage import LocalExperiment

from .everest_ensemble import EverestEnsemble


class EverestExperiment:
    def __init__(self, ert_experiment: LocalExperiment) -> None:
        self._ert_experiment = ert_experiment

    @cached_property
    def ensembles(self) -> list[EverestEnsemble]:
        return [
            EverestEnsemble(ens)
            for ens in self._ert_experiment._storage.ensembles
            if ens.experiment_id == self._ert_experiment.id
        ]

    def create_ensemble(
        self,
        ensemble_size: int,
        name: str,
    ) -> EverestEnsemble:
        ert_ensemble = self._ert_experiment.create_ensemble(
            ensemble_size=ensemble_size, name=name
        )
        everest_ensemble = EverestEnsemble(ert_ensemble)
        if self.ensembles is not None:
            del self.ensembles  # Clear cache when a new ensemble is created
        return everest_ensemble
