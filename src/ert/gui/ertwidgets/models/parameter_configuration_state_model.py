from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal as Signal

from ert.config.parameter_config import LocalizationType, ParameterConfig


@dataclass
class _ParameterConfigurations:
    init: list[ParameterConfig]
    draft: list[ParameterConfig] | None = None

    @property
    def current(self) -> list[ParameterConfig]:
        return self.init if self.draft is None else self.draft


class ParameterConfigurationStateModel(QObject):
    changed = Signal()

    def __init__(
        self,
        parameters: Iterable[ParameterConfig],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._configured_parameters = _ParameterConfigurations(list(parameters))
        self._prior_parameters: _ParameterConfigurations | None = None

    @property
    def _active(self) -> _ParameterConfigurations:
        return self._prior_parameters or self._configured_parameters

    @property
    def parameters(self) -> list[ParameterConfig]:
        return self._active.current

    @property
    def update_strategies(self) -> dict[str, LocalizationType]:
        return {
            parameter.type.upper(): parameter.update_strategy
            for parameter in self.parameters
            if parameter.update_strategy is not None
        }

    def apply_update_strategies(
        self, update_strategies: Mapping[str, LocalizationType]
    ) -> None:
        active = self._active
        active.draft = [
            parameter.model_copy(
                update={"update_strategy": update_strategies[parameter.type.upper()]}
            )
            if parameter.update_strategy is not None
            and parameter.type.upper() in update_strategies
            else parameter
            for parameter in active.current
        ]
        self.changed.emit()

    def reset_parameters(self) -> None:
        self._configured_parameters.draft = None
        if self._prior_parameters:
            self._prior_parameters.draft = None
        self.changed.emit()

    def select_prior(self, prior_parameter_config: Iterable[ParameterConfig]) -> None:
        prior = list(prior_parameter_config)
        if self._prior_parameters is None or self._prior_parameters.init != prior:
            self._prior_parameters = _ParameterConfigurations(prior)
        self.changed.emit()

    def deselect_prior(self) -> None:
        self._prior_parameters = None
        self.changed.emit()
