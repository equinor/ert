from collections.abc import Iterable, Mapping

from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal as Signal

from ert.config.parameter_config import (
    LocalizationType,
    ParameterConfig,
    ParameterType,
)


class ParameterConfigurationStateModel(QObject):
    changed = Signal()

    def __init__(
        self,
        parameters: Iterable[ParameterConfig],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_configured_parameters = list(parameters)
        self._draft_configured_parameters: list[ParameterConfig] | None = None
        self._prior_parameters: list[ParameterConfig] | None = None

    @property
    def parameters(self) -> list[ParameterConfig]:
        if self._prior_parameters is not None:
            return self._prior_parameters
        return self._configured_parameters

    @property
    def has_prior(self) -> bool:
        return self._prior_parameters is not None

    @property
    def _configured_parameters(self) -> list[ParameterConfig]:
        if self._draft_configured_parameters is not None:
            return self._draft_configured_parameters
        return self._init_configured_parameters

    @property
    def update_strategies(self) -> dict[ParameterType, LocalizationType]:
        return {
            parameter.type: parameter.update_strategy
            for parameter in self.parameters
            if parameter.update_strategy is not None
            and isinstance(parameter.type, ParameterType)
        }

    def apply_update_strategies(
        self, update_strategies: Mapping[ParameterType, LocalizationType]
    ) -> None:
        if self.has_prior:
            raise RuntimeError("Parameters from prior ensemble cannot be changed.")

        self._draft_configured_parameters = [
            parameter.model_copy(
                update={"update_strategy": update_strategies[parameter.type]}
            )
            if parameter.update_strategy is not None
            and isinstance(parameter.type, ParameterType)
            and parameter.type in update_strategies
            else parameter
            for parameter in self._configured_parameters
        ]
        self.changed.emit()

    def select_prior(self, prior_parameter_config: Iterable[ParameterConfig]) -> None:
        self._prior_parameters = list(prior_parameter_config)
        self.changed.emit()

    def deselect_prior(self) -> None:
        self._prior_parameters = None
        self.changed.emit()
