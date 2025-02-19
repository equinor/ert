from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

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

    def find_realization_with_data(
        self,
        parameter_values: dict[str, list[float]],
        exclude: Iterable[int] | None = None,
    ) -> tuple[int, EverestEnsemble] | tuple[None, None]:
        if not list(self.ensembles):
            return None, None

        if exclude is None:
            exclude = []

        for e in self.ensembles:
            ens_parameters = {
                group: e.ert_ensemble.load_parameters(group)
                .to_dataarray()
                .data.reshape((e.ert_ensemble.ensemble_size, -1))
                for group in parameter_values
            }

            matching_real = next(
                (
                    i
                    for i in range(e.ert_ensemble.ensemble_size)
                    if i not in exclude
                    and all(
                        np.allclose(ens_parameters[group][i], group_data)
                        for group, group_data in parameter_values.items()
                    )
                ),
                None,
            )

            if matching_real is not None:
                return matching_real, e

        return None, None
