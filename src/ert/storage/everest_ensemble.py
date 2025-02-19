from pydantic import BaseModel
from typing_extensions import TypedDict

from .local_ensemble import LocalEnsemble


class EverestRealizationInfo(TypedDict):
    model_realization: int
    perturbation: int | None  # None means it stems from unperturbed controls


class _Index(BaseModel):
    ert2ev_realization_mapping: dict[int, EverestRealizationInfo] | None = None


class EverestEnsemble:
    def __init__(self, ert_ensemble: LocalEnsemble):
        self._ert_ensemble = ert_ensemble
        self._index = _Index()

    @property
    def ert_ensemble(self) -> LocalEnsemble:
        return self._ert_ensemble

    def save_realization_mapping(
        self, realization_mapping: dict[int, EverestRealizationInfo]
    ) -> None:
        self._index.ert2ev_realization_mapping = realization_mapping
        self._ert_ensemble._storage._write_transaction(
            self.ert_ensemble._path / "everest_index.json",
            self._index.model_dump_json().encode("utf-8"),
        )
        self._index = _Index.model_validate_json(
            (self.ert_ensemble._path / "everest_index.json").read_text(encoding="utf-8")
        )
