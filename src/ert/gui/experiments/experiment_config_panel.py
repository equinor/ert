from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ert.gui.ertwidgets import CopyableLabel
from ert.gui.ertwidgets.accordion import Accordion
from ert.gui.ertwidgets.card import CARD_SPACING, create_card, create_card_title

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

    A double width card on the left holds the settings common to all
    experiment types (see :meth:`add_common_settings`), and a card to its
    right holds the "Options" and "Update settings" accordions.

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

    def add_common_settings(
        self,
        runpath: str,
        number_of_realizations: QWidget,
        active_realizations: QWidget,
        experiment_name: QWidget | None = None,
    ) -> None:
        self.common_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        # Center labels on their fields so label and value text share a baseline.
        self.common_layout.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        if experiment_name is not None:
            self.common_layout.addRow("Experiment name:", experiment_name)
        self.common_layout.addRow("Runpath:", CopyableLabel(text=runpath, elide=True))
        self.common_layout.addRow("Number of realizations:", number_of_realizations)
        self.common_layout.addRow("Active realizations:", active_realizations)

    def build_layout(self, *, accordions: bool = True) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(CARD_SPACING)
        for column in range(3):
            grid.setColumnStretch(column, 1)

        main_card = create_card()
        main_layout = QVBoxLayout(main_card)
        main_layout.setSpacing(0)
        if accordions:
            main_layout.setContentsMargins(0, 0, 0, 0)
            self.options_accordion = Accordion(
                "Options", self._wrap(self.options_layout), expanded=True
            )
            self.options_accordion.setObjectName("options_accordion")
            main_layout.addWidget(self.options_accordion)
            self.update_settings_accordion = Accordion(
                "Update settings",
                self._wrap(self.update_settings_layout),
                expanded=False,
            )
            self.update_settings_accordion.setObjectName("update_settings_accordion")
            self.update_settings_accordion.setVisible(
                self.update_settings_layout.rowCount() > 0
            )
            main_layout.addWidget(self.update_settings_accordion)
        else:
            main_layout.setContentsMargins(16, 12, 16, 12)
            main_layout.addWidget(self._wrap(self.options_layout))
        main_layout.addStretch(1)
        grid.addWidget(main_card, 0, 2)

        if self.common_layout.rowCount() > 0:
            common_card = create_card()
            common_layout = QVBoxLayout(common_card)
            common_layout.setContentsMargins(16, 12, 16, 12)
            common_layout.addWidget(create_card_title("Common settings"))
            common_layout.addWidget(self._wrap(self.common_layout))
            common_layout.addStretch(1)
            grid.addWidget(common_card, 0, 0, 1, 2)
            # QGridLayout drops the spacing next to a column holding only a
            # spanned item; an invisible widget in that column keeps the three
            # columns aligned with the summary cards below.
            column_anchor = QWidget()
            column_anchor.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
            )
            column_anchor.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            grid.addWidget(column_anchor, 0, 1)
            column_anchor.lower()

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
