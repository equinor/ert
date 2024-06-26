from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator, List, Optional, Union

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtGui import QIcon, QKeyEvent
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ert.gui.plottery import PlotConfig, PlotConfigFactory, PlotConfigHistory
from ert.gui.tools.plot.plot_api import PlotApiKeyDefinition
from ert.gui.tools.plot.widgets import CopyStyleToDialog

from .default_customization_view import DefaultCustomizationView
from .limits_customization_view import LimitsCustomizationView
from .statistics_customization_view import StatisticsCustomizationView
from .style_customization_view import StyleCustomizationView

if TYPE_CHECKING:
    from ert.gui.tools.plot.customize import CustomizationView


class PlotCustomizer(QObject):
    settingsChanged = Signal()

    def __init__(
        self, parent: Optional[QWidget], key_defs: List[PlotApiKeyDefinition]
    ) -> None:
        super().__init__()

        self._plot_config_key = None
        self._previous_key = None
        self.default_plot_settings = None
        self._plot_configs: dict[Optional[str], PlotConfigHistory] = {
            None: PlotConfigHistory(
                "No_Key_Selected", PlotConfig(plot_settings=None, title=None)
            )
        }

        self._customization_dialog = CustomizePlotDialog(
            "Customize", parent, key_defs, key=self._plot_config_key
        )

        self._customization_dialog.addTab(
            "general", "General", DefaultCustomizationView()
        )
        self._customization_dialog.addTab("style", "Style", StyleCustomizationView())
        self._customization_dialog.addTab(
            "statistics", "Statistics", StatisticsCustomizationView()
        )

        self._customize_limits = LimitsCustomizationView()
        self._customization_dialog.addTab("limits", "Limits", self._customize_limits)

        self._customization_dialog.applySettings.connect(self.applyCustomization)
        self._customization_dialog.undoSettings.connect(self.undoCustomization)
        self._customization_dialog.redoSettings.connect(self.redoCustomization)
        self._customization_dialog.resetSettings.connect(self.resetCustomization)
        self._customization_dialog.copySettings.connect(self.copyCustomization)
        self._customization_dialog.copySettingsToOthers.connect(
            self.copyCustomizationTo
        )
        self._revertCustomization(self.getPlotConfig())

    def _getPlotConfigHistory(self) -> PlotConfigHistory:
        return self._plot_configs[self._plot_config_key]

    def undoCustomization(self) -> None:
        history = self._getPlotConfigHistory()
        history.undoChanges()
        self._revertCustomization(history.getPlotConfig())

    def redoCustomization(self) -> None:
        history = self._getPlotConfigHistory()
        history.redoChanges()
        self._revertCustomization(history.getPlotConfig())

    def resetCustomization(self) -> None:
        history = self._getPlotConfigHistory()
        history.resetChanges()
        self._revertCustomization(history.getPlotConfig())

    def applyCustomization(self) -> None:
        history = self._getPlotConfigHistory()
        plot_config = history.getPlotConfig()
        if self._customization_dialog is not None:
            for customization_view in self._customization_dialog:
                customization_view.applyCustomization(plot_config)

        history.applyChanges(plot_config)

        self._emitChangedSignal()

    def _revertCustomization(self, plot_config: PlotConfig, emit: bool = True) -> None:
        if self._customization_dialog is not None:
            for customization_view in self._customization_dialog:
                customization_view.revertCustomization(plot_config)

        self._emitChangedSignal(emit)

    def _emitChangedSignal(self, emit: bool = True) -> None:
        history = self._getPlotConfigHistory()
        self._customization_dialog.setUndoRedoCopyState(
            history.isUndoPossible(), history.isRedoPossible(), self.isCopyPossible()
        )

        if emit:
            self.settingsChanged.emit()

    def isCopyPossible(self) -> bool:
        return len(self._plot_configs) > 2

    def copyCustomizationTo(self, keys: Iterable[str]) -> None:
        """copies the plotconfig of the current key, to a set of other keys"""
        history = self._getPlotConfigHistory()

        for key in keys:
            if key not in self._plot_configs:
                self._plot_configs[key] = PlotConfigHistory(
                    "No_Key_Selected",
                    PlotConfig(self.default_plot_settings, title=None),
                )
            source_config = history.getPlotConfig()
            source_config.setTitle(key)

            self._plot_configs[key].applyChanges(source_config)

            self._customization_dialog.addCopyableKey(key)

        self._emitChangedSignal(emit=True)

    def copyCustomization(self, key: Optional[str]) -> None:
        key = str(key)
        if self.isCopyPossible():
            source_config = self._plot_configs[key].getPlotConfig()
            source_config.setTitle(None)

            history = self._getPlotConfigHistory()
            history.applyChanges(source_config)

            self._revertCustomization(history.getPlotConfig())

    def toggleCustomizationDialog(self) -> None:
        if self._customization_dialog.isVisible():
            self._customization_dialog.hide()
        else:
            self._customization_dialog.show()

    def switchPlotConfigHistory(self, key_def: PlotApiKeyDefinition) -> None:
        if key_def is None:
            return
        key = key_def.key
        if key != self._plot_config_key:
            if key not in self._plot_configs:
                self._plot_configs[key] = PlotConfigHistory(
                    key, PlotConfigFactory.createPlotConfigForKey(key_def)
                )
                self._customization_dialog.addCopyableKey(key)
            self._customization_dialog.currentPlotKeyChanged(key)
            self._previous_key = self._plot_config_key
            self._plot_config_key = key
            self._revertCustomization(self.getPlotConfig(), emit=False)

    def getPlotConfig(self) -> PlotConfig:
        return self._getPlotConfigHistory().getPlotConfig()

    def setAxisTypes(
        self, x_axis_type: Optional[str], y_axis_type: Optional[str]
    ) -> None:
        self._customize_limits.setAxisTypes(x_axis_type, y_axis_type)


class CustomizePlotDialog(QDialog):
    applySettings = Signal()
    undoSettings = Signal()
    redoSettings = Signal()
    resetSettings = Signal()
    copySettings = Signal(str)
    copySettingsToOthers = Signal(list)

    def __init__(
        self,
        title: Optional[str],
        parent: Optional[QWidget],
        key_defs: List[PlotApiKeyDefinition],
        key: Optional[str] = "",
    ) -> None:
        QDialog.__init__(self, parent)
        self.setWindowTitle(title)

        self.current_key = key
        self._key_defs = key_defs

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # type: ignore
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)  # type: ignore

        self._tab_map: dict[str, QWidget] = {}
        self._tab_order: list[str] = []

        layout = QVBoxLayout()

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        self._button_layout = QHBoxLayout()

        self._reset_button = QToolButton()
        self._reset_button.setIcon(QIcon("img:format_color_reset.svg"))
        self._reset_button.setToolTip("Reset all settings back to default")
        self._reset_button.clicked.connect(self.resetSettings)

        self._undo_button = QToolButton()
        self._undo_button.setIcon(QIcon("img:undo.svg"))
        self._undo_button.setToolTip("Undo")
        self._undo_button.clicked.connect(self.undoSettings)

        self._redo_button = QToolButton()
        self._redo_button.setIcon(QIcon("img:redo.svg"))
        self._redo_button.setToolTip("Redo")
        self._redo_button.clicked.connect(self.redoSettings)
        self._redo_button.setEnabled(False)

        self._copy_from_button = QToolButton()
        self._copy_from_button.setIcon(QIcon("img:download.svg"))
        self._copy_from_button.setToolTip("Copy settings from another key")
        self._copy_from_button.setPopupMode(QToolButton.InstantPopup)
        self._copy_from_button.setEnabled(False)

        self._copy_to_button = QToolButton()
        self._copy_to_button.setIcon(QIcon("img:upload.svg"))
        self._copy_to_button.setToolTip("Copy current plot settings to other keys")
        self._copy_to_button.setPopupMode(QToolButton.InstantPopup)
        self._copy_to_button.clicked.connect(self.initiateCopyStyleToDialog)
        self._copy_to_button.setEnabled(True)

        tool_menu = QMenu(self._copy_from_button)
        self._popup_list = QListWidget(tool_menu)
        self._popup_list.setSortingEnabled(True)
        self._popup_list.itemClicked.connect(self.keySelected)
        action = QWidgetAction(tool_menu)
        action.setDefaultWidget(self._popup_list)
        tool_menu.addAction(action)
        self._copy_from_button.setMenu(tool_menu)

        self._apply_button = QPushButton("Apply")
        self._apply_button.setToolTip("Apply the new settings")
        self._apply_button.clicked.connect(self.applySettings)
        self._apply_button.setDefault(True)

        self._close_button = QPushButton("Close")
        self._close_button.setToolTip("Hide this dialog")
        self._close_button.clicked.connect(self.hide)

        self._button_layout.addWidget(self._reset_button)
        self._button_layout.addStretch()
        self._button_layout.addWidget(self._undo_button)
        self._button_layout.addWidget(self._redo_button)
        self._button_layout.addWidget(self._copy_from_button)
        self._button_layout.addWidget(self._copy_to_button)
        self._button_layout.addStretch()
        self._button_layout.addWidget(self._apply_button)
        self._button_layout.addWidget(self._close_button)

        layout.addStretch()
        layout.addLayout(self._button_layout)

        self.setLayout(layout)

    def initiateCopyStyleToDialog(self) -> None:
        dialog = CopyStyleToDialog(self, self.current_key, self._key_defs)
        if dialog.exec_():
            self.copySettingsToOthers.emit(dialog.getSelectedKeys())

    def addCopyableKey(self, key: Optional[Union[str, QListWidgetItem]]) -> None:
        self._popup_list.addItem(key)

    def keySelected(self, list_widget_item: QListWidgetItem) -> None:
        self.copySettings.emit(str(list_widget_item.text()))

    def currentPlotKeyChanged(self, new_key: Optional[str]) -> None:
        self.current_key = new_key

    def keyPressEvent(self, a0: Optional[QKeyEvent]) -> None:
        # Hide when pressing Escape instead of QDialog.keyPressEvent(KeyEscape)
        # which closes the dialog
        if a0 is not None and a0.key() == Qt.Key.Key_Escape:
            self.hide()
        else:
            QDialog.keyPressEvent(self, a0)

    def addTab(self, attribute_name: str, title: str, widget: QWidget) -> None:
        self._tabs.addTab(widget, title)
        self._tab_map[attribute_name] = widget
        self._tab_order.append(attribute_name)

    def __getitem__(self, item: str) -> "CustomizationView":
        return self._tab_map[item]

    def __iter__(self) -> Iterator[QWidget]:
        for attribute_name in self._tab_order:
            yield self._tab_map[attribute_name]

    def setUndoRedoCopyState(self, undo: bool, redo: bool, copy: bool = False) -> None:
        self._undo_button.setEnabled(undo)
        self._redo_button.setEnabled(redo)
        self._copy_from_button.setEnabled(copy)
