from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QVBoxLayout,
    QWidget,
)

from ert.gui.ertwidgets import CopyableLabel
from ert.gui.ertwidgets.accordion import Accordion
from ert.gui.ertwidgets.card import Card

if TYPE_CHECKING:
    from ert.config import ParameterConfig
    from ert.run_models import RunModel
    from ert.storage import Ensemble


def _create_form() -> QFormLayout:
    form = QFormLayout()
    form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    form.setContentsMargins(0, 0, 0, 0)
    return form


class ExperimentConfigPanel(QWidget):
    """Settings for one experiment type, laid out as a grid of cards.

    The first card holds the settings common to all experiment types (see
    :meth:`add_common_settings`), followed by cards for the "{mode} settings"
    and the "Update settings".

    Subclasses add rows to ``options_layout`` and ``update_settings_layout``
    and call :meth:`build_layout` when done. To keep the options in the same
    order across experiment types, add them in this order (skipping the ones
    that do not apply):

    ensemble name, update method, source ensemble, prior ensemble, ensemble
    format, weights, design matrix.
    """

    experiment_configuration_changed = Signal()

    def __init__(self, run_model: type[RunModel]) -> None:
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.__run_model = run_model
        self._parameter_configuration_getter: (
            Callable[[], list[ParameterConfig]] | None
        ) = None

        self.options_layout = _create_form()
        self.update_settings_layout = _create_form()
        self.common_layout = _create_form()
        self.options_accordion: Accordion | None = None
        self.update_settings_accordion: Accordion | None = None
        self.update_settings_card: QWidget | None = None
        self.common_accordion: Accordion | None = None

    def add_common_settings(
        self,
        runpath: str,
        number_of_realizations: QWidget,
        active_realizations: QWidget,
        experiment_name: QWidget,
    ) -> None:
        self.common_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        # Labels default to top alignment, which misaligns them with taller fields.
        self.common_layout.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.common_layout.addRow("Experiment name:", experiment_name)
        self.common_layout.addRow("Runpath:", CopyableLabel(text=runpath, wrap=True))
        self.common_layout.addRow("Number of realizations:", number_of_realizations)
        self.common_layout.addRow("Active realizations:", active_realizations)

    def build_layout(self, *, accordions: bool = True) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(Card.SPACING)
        for column in range(3):
            grid.setColumnStretch(column, 1)

        column = 0
        if self.common_layout.rowCount() > 0:
            self.common_accordion = Accordion(
                "Main settings", self._wrap(self.common_layout), expanded=True
            )
            self.common_accordion.setObjectName("common_accordion")
            grid.addWidget(self._accordion_card(self.common_accordion), 0, column)
            column += 1

        if accordions:
            self.options_accordion = Accordion(
                f"{self.__run_model.name()} settings",
                self._wrap(self.options_layout),
                expanded=True,
            )
            self.options_accordion.setObjectName("options_accordion")
            grid.addWidget(self._accordion_card(self.options_accordion), 0, column)
            column += 1
            self.update_settings_accordion = Accordion(
                "Update settings",
                self._wrap(self.update_settings_layout),
                expanded=False,
            )
            self.update_settings_accordion.setObjectName("update_settings_accordion")
            self.update_settings_card = self._accordion_card(
                self.update_settings_accordion
            )
            self.update_settings_card.setVisible(
                self.update_settings_layout.rowCount() > 0
            )
            grid.addWidget(self.update_settings_card, 0, column)
        else:
            options_card = Card()
            options_layout = QVBoxLayout(options_card)
            options_layout.setContentsMargins(16, 12, 16, 12)
            options_layout.addWidget(self._wrap(self.options_layout))
            options_layout.addStretch(1)
            grid.addWidget(options_card, 0, column)

    @staticmethod
    def _accordion_card(accordion: Accordion) -> QWidget:
        card = Card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)
        card_layout.addWidget(accordion)
        card_layout.addStretch(1)
        return card

    @staticmethod
    def _wrap(layout: QFormLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

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

    def get_selected_ensemble(self) -> Ensemble | None:
        """The ensemble whose stored parameters and observations the
        experiment uses instead of those in the config file, if any.
        """
        return None

    def isConfigurationValid(self) -> bool:
        return True

    def get_experiment_arguments(self) -> Any:
        return {}

    @Slot(QWidget)
    def experimentTypeChanged(self, w: QWidget) -> Any:
        pass
