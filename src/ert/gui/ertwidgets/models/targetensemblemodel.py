from typing import override

from ert.config import AnalysisConfig
from ert.gui.ertnotifier import ErtNotifier

from .valuemodel import ValueModel


class TargetEnsembleModel(ValueModel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        notifier: ErtNotifier,
    ) -> None:
        self.analysis_config = analysis_config
        self.notifier = notifier
        self._custom = False
        super().__init__(self.getDefaultValue())
        notifier.ertChanged.connect(self._on_current_ensemble_changed)

    @override
    def setValue(self, value: str | None) -> None:
        """Set a new target ensemble"""
        if value == self.getDefaultValue():
            self._custom = False
            ValueModel.setValue(self, self.getDefaultValue())
        else:
            self._custom = True
            ValueModel.setValue(self, value)

    def getDefaultValue(self) -> str | None:
        return "iter-%d"

    def _on_current_ensemble_changed(self) -> None:
        if not self._custom:
            super().setValue(self.getDefaultValue())
