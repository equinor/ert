from collections.abc import Iterable

from PyQt6.QtCore import QObject, pyqtSignal

from ert.config import LocalizationType, ParameterConfig
from ert.config.parameter_config import apply_parameter_update_overrides


class ParameterConfiguration(QObject):
    changed = pyqtSignal()

    def __init__(self, parameters: Iterable[ParameterConfig]) -> None:
        super().__init__()
        self._configured = [p.model_copy(deep=True) for p in parameters]
        self._configured_overrides: dict[str, LocalizationType] = {}
        self._prior: list[ParameterConfig] = []
        self._prior_overrides: dict[str, LocalizationType] = {}
        self._prior_selected = False
        self._prior_id: str | None = None
        self.revision = 0

    @property
    def available(self) -> bool:
        return not self._prior_selected or self._prior_id is not None

    @property
    def prior_id(self) -> str | None:
        return self._prior_id

    @property
    def parameters(self) -> list[ParameterConfig]:
        return apply_parameter_update_overrides(self._baseline, self.overrides)

    @property
    def _baseline(self) -> list[ParameterConfig]:
        return self._prior if self._prior_selected else self._configured

    @property
    def overrides(self) -> dict[str, LocalizationType]:
        return dict(
            self._prior_overrides
            if self._prior_selected
            else self._configured_overrides
        )

    @property
    def has_changes(self) -> bool:
        return bool(self._configured_overrides or self._prior_overrides)

    def select_prior(
        self,
        selected: bool,
        prior_id: str | None = None,
        parameters: Iterable[ParameterConfig] = (),
    ) -> None:
        prior = list(parameters) if selected and prior_id is not None else []
        if (selected, prior_id) == (
            self._prior_selected,
            self._prior_id,
        ) and prior == self._prior:
            return
        self._prior_selected = selected
        self._prior_id = prior_id
        self._prior = [p.model_copy(deep=True) for p in prior]
        self._prior_overrides = {}
        self._notify_changed()

    def apply_strategies(self, strategies: dict[str, LocalizationType]) -> None:
        overrides = self.overrides
        for parameter in self._baseline:
            if parameter.update_strategy is None or parameter.type not in strategies:
                continue
            strategy = strategies[parameter.type]
            if strategy == parameter.update_strategy:
                overrides.pop(parameter.name, None)
            else:
                overrides[parameter.name] = strategy
        apply_parameter_update_overrides(self._baseline, overrides)
        if overrides == self.overrides:
            return
        if self._prior_selected:
            self._prior_overrides = overrides
        else:
            self._configured_overrides = overrides
        self._notify_changed()

    def reset(self) -> None:
        if not self.has_changes:
            return
        self._configured_overrides.clear()
        self._prior_overrides.clear()
        self._notify_changed()

    def _notify_changed(self) -> None:
        self.revision += 1
        self.changed.emit()
