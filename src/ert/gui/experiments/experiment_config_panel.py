from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from ert.config import ParameterConfig
    from ert.run_models import RunModel


class ExperimentConfigPanel(QWidget):
    experiment_configuration_changed = Signal()

    def __init__(self, run_model: type[RunModel]) -> None:
        super().__init__()
        self.setContentsMargins(10, 10, 10, 10)
        self.__run_model = run_model
        self._parameter_configuration_getter: (
            Callable[[], list[ParameterConfig]] | None
        ) = None

    def get_experiment_type(self) -> type[RunModel]:
        return self.__run_model

    def set_parameter_configuration_getter(
        self, getter: Callable[[], list[ParameterConfig]]
    ) -> None:
        self._parameter_configuration_getter = getter

    def get_parameter_configuration(self) -> list[ParameterConfig] | None:
        """The parameters this experiment will use, or None if the panel
        does not show them.
        """
        if self._parameter_configuration_getter is None:
            return None
        return self._parameter_configuration_getter()

    def isConfigurationValid(self) -> bool:
        return True

    def get_experiment_arguments(self) -> Any:
        return {}

    @Slot(QWidget)
    def experimentTypeChanged(self, w: QWidget) -> Any:
        pass
