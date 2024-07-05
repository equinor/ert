from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon

from ert.gui.tools import Tool
from ert.gui.tools.manage_experiments import ManageExperimentsPanel

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.gui.ertnotifier import ErtNotifier


class ManageExperimentsTool(Tool):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier, ensemble_size: int):
        super().__init__("Manage experiments", QIcon("img:build_wrench.svg"))
        self.notifier = notifier
        self.ert_config = config
        self.ensemble_size = ensemble_size
        self._manage_experiments_panel: Optional[ManageExperimentsPanel] = None

    def trigger(self) -> None:
        if not self._manage_experiments_panel:
            self._manage_experiments_panel = ManageExperimentsPanel(
                self.ert_config, self.notifier, self.ensemble_size
            )
            self._manage_experiments_panel.setWindowModality(
                Qt.WindowModality.ApplicationModal
            )

        self._manage_experiments_panel.show()
