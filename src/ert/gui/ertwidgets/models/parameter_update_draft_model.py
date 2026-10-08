from collections.abc import Iterable, Mapping

from PyQt6.QtCore import QObject, pyqtSignal

from ert.config import LocalizationType, ParameterConfig
from ert.parameter_update_draft import ParameterUpdateDraft


class ParameterUpdateDraftModel(QObject):
    changed = pyqtSignal()

    def __init__(self, parameters: Iterable[ParameterConfig]) -> None:
        super().__init__()
        self._draft = ParameterUpdateDraft(parameters)
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def has_parameter_source(self) -> bool:
        return self._draft.has_parameter_source

    @property
    def prior_id(self) -> str | None:
        return self._draft.prior_id

    @property
    def parameters(self) -> list[ParameterConfig]:
        return self._draft.parameters

    @property
    def overrides(self) -> dict[str, LocalizationType]:
        return self._draft.overrides

    @property
    def has_changes(self) -> bool:
        return self._draft.has_changes

    def select_prior(
        self,
        prior_id: str | None,
        parameters: Iterable[ParameterConfig],
    ) -> None:
        if self._draft.select_prior(prior_id, parameters):
            self._notify_changed()

    def use_configured_parameters(self) -> None:
        if self._draft.use_configured_parameters():
            self._notify_changed()

    def apply_strategies_by_type(
        self, strategies: Mapping[str, LocalizationType]
    ) -> None:
        if self._draft.apply_strategies_by_type(strategies):
            self._notify_changed()

    def reset_all_overrides(self) -> None:
        if self._draft.reset_all_overrides():
            self._notify_changed()

    def _notify_changed(self) -> None:
        self._revision += 1
        self.changed.emit()
