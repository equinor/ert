from typing import Any, Optional

from ert.config import AnalysisConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.models.valuemodel import ValueModel


class TargetEnsembleModel(ValueModel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        notifier: ErtNotifier,
    ):
        self.analysis_config = analysis_config
        self.notifier = notifier
        self._custom = False
        super().__init__(self.getDefaultValue())
        notifier.ertChanged.connect(self.on_current_ensemble_changed)
        notifier.current_ensemble_changed.connect(self.on_current_ensemble_changed)

    def setValue(self, value: Optional[str]) -> None:
        """Set a new target ensemble"""
        if value == self.getDefaultValue():
            self._custom = False
            ValueModel.setValue(self, self.getDefaultValue())
        else:
            self._custom = True
            ValueModel.setValue(self, value)

    def getDefaultValue(self) -> Optional[str]:
        ensemble_name = self.notifier.current_ensemble_name
        return f"{ensemble_name}_%d"

    def on_current_ensemble_changed(self, *args: Any) -> None:
        if not self._custom:
            super().setValue(self.getDefaultValue())
