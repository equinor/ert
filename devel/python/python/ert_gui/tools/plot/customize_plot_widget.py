from PyQt4.QtCore import pyqtSignal
from PyQt4.QtGui import QWidget, QVBoxLayout, QCheckBox
from ert_gui.models.mixins.connectorless.default_choice_list_model import DefaultChoiceListModel

from ert_gui.tools.plot import ColorChooser
from ert_gui.tools.plot.style_chooser import StyleChooser
from ert_gui.widgets.combo_choice import ComboChoice


class CustomizePlotWidget(QWidget):

    customPlotSettingsChanged = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)
        self.__custom = { }
        self.__style_choosers = {}

        self.__layout = QVBoxLayout()

        self.addCheckBox("show_observations", "Show observations", True)
        # self.addCheckBox("show_refcase", "Show refcase", True)
        self.addCheckBox("show_legend", "Show legend", True)
        self.addCheckBox("show_grid", "Show grid", True)
        self.addCheckBox("show_distribution_lines", "Show distribution lines", False)

        self.__layout.addSpacing(10)
        self.addStyleChooser("default_style", "Default", StyleChooser.STYLE_SOLID, StyleChooser.MARKER_OFF, labeled=True)
        self.addStyleChooser("refcase_style", "Refcase", StyleChooser.STYLE_SOLID, StyleChooser.MARKER_OFF)

        self.__layout.addSpacing(20)
        self.addPresets("Presets", ["Default", "Overview", "All statistics"], self.applyPreset)

        self.__layout.addSpacing(5)
        self.addStyleChooser("mean_style", "Mean", StyleChooser.STYLE_SOLID, StyleChooser.MARKER_OFF)
        self.addStyleChooser("p50_style", "P50", StyleChooser.STYLE_OFF, StyleChooser.MARKER_OFF)
        self.addStyleChooser("min-max_style", "Min/Max", StyleChooser.STYLE_OFF, StyleChooser.MARKER_OFF, True)
        self.addStyleChooser("p10-p90_style", "P10-P90", StyleChooser.STYLE_DASHED, StyleChooser.MARKER_OFF, True)
        self.addStyleChooser("p33-p67_style", "P33-P67", StyleChooser.STYLE_OFF, StyleChooser.MARKER_OFF, True)

        # self.addColorChooser("observation", "Observation", QColor(0, 0, 0, 255))
        # self.addColorChooser("observation_area", "Observation Error", QColor(0, 0, 0, 38))
        # self.addColorChooser("observation_error_bar", "Observation Error Bar", QColor(0, 0, 0, 255))
        # self.addColorChooser("refcase", "Refcase", QColor(0, 0, 0, 178))
        # self.addColorChooser("ensemble_1", "Case #1", QColor(56, 108, 176, 204))
        # self.addColorChooser("ensemble_2", "Case #2", QColor(127, 201, 127, 204))
        # self.addColorChooser("ensemble_3", "Case #3", QColor(253, 192, 134, 204))
        # self.addColorChooser("ensemble_4", "Case #4", QColor(240, 2, 127, 204))
        # self.addColorChooser("ensemble_5", "Case #5", QColor(191, 91, 23, 204))

        self.__layout.addStretch()

        self.setLayout(self.__layout)

    def emitChange(self):
        self.customPlotSettingsChanged.emit()

    def addCheckBox(self, name, description, default_value):
        checkbox = QCheckBox(description)
        checkbox.setChecked(default_value)
        self.__custom[name] = default_value

        def toggle(checked):
            self.__custom[name] = checked
            self.emitChange()

        checkbox.toggled.connect(toggle)

        self.__layout.addWidget(checkbox)

    def createJSColor(self, color):
        return "rgba(%d, %d, %d, %f)" % (color.red(), color.green(), color.blue(), color.alphaF())

    def getCustomSettings(self):
        return self.__custom
    
    def addColorChooser(self, name, label, default_color):
        color_chooser = ColorChooser(label, default_color)
        self.__custom[name] = self.createJSColor(default_color)

        def colorChanged(color):
            self.__custom[name] = self.createJSColor(color)
            self.emitChange()

        color_chooser.colorChanged.connect(colorChanged)
        self.__layout.addWidget(color_chooser)


    def addStyleChooser(self, name, description, line_style=StyleChooser.STYLE_OFF, marker_style=StyleChooser.MARKER_OFF, area_supported=False, labeled=False):
        style_chooser = StyleChooser(description, line_style, marker_style, area_supported, labeled)
        self.__style_choosers[name] = style_chooser
        self.__custom[name] = (line_style[1], marker_style[1]) # not pretty

        def styleChanged(line, marker):
            self.__custom[name] = (str(line), str(marker))
            self.emitChange()

        style_chooser.styleChanged.connect(styleChanged)
        self.__layout.addWidget(style_chooser)

    def addPresets(self, name, presets, presetFunction):
        model = DefaultChoiceListModel(presets)
        combo_choice = ComboChoice(model, name)
        combo_choice.includeLabel()

        def selectionChanged():
            current = model.getCurrentChoice()
            presetFunction(current)

        model.observable().attach(model.CURRENT_CHOICE_CHANGED_EVENT, selectionChanged)

        self.__layout.addWidget(combo_choice)

    def applyPreset(self, preset_name):
        blocked = self.signalsBlocked()
        self.blockSignals(True)
        print("Applying preset for %s" % preset_name)

        if preset_name == "Default":
            self.__style_choosers["mean_style"].updateLineStyleAndMarker(StyleChooser.STYLE_SOLID, StyleChooser.MARKER_OFF)
            self.__style_choosers["p50_style"].updateLineStyleAndMarker(StyleChooser.STYLE_OFF, StyleChooser.MARKER_OFF)
            self.__style_choosers["min-max_style"].updateLineStyleAndMarker(StyleChooser.STYLE_OFF, StyleChooser.MARKER_OFF)
            self.__style_choosers["p10-p90_style"].updateLineStyleAndMarker(StyleChooser.STYLE_DASHED, StyleChooser.MARKER_OFF)
            self.__style_choosers["p33-p67_style"].updateLineStyleAndMarker(StyleChooser.STYLE_OFF, StyleChooser.MARKER_OFF)

        elif preset_name ==  "Overview":
            self.__style_choosers["mean_style"].updateLineStyleAndMarker(StyleChooser.STYLE_OFF, StyleChooser.MARKER_OFF)
            self.__style_choosers["p50_style"].updateLineStyleAndMarker(StyleChooser.STYLE_OFF, StyleChooser.MARKER_OFF)
            self.__style_choosers["min-max_style"].updateLineStyleAndMarker(StyleChooser.STYLE_AREA, StyleChooser.MARKER_OFF)
            self.__style_choosers["p10-p90_style"].updateLineStyleAndMarker(StyleChooser.STYLE_OFF, StyleChooser.MARKER_OFF)
            self.__style_choosers["p33-p67_style"].updateLineStyleAndMarker(StyleChooser.STYLE_OFF, StyleChooser.MARKER_OFF)

        elif preset_name == "All statistics":
            self.__style_choosers["mean_style"].updateLineStyleAndMarker(StyleChooser.STYLE_SOLID, StyleChooser.MARKER_OFF)
            self.__style_choosers["p50_style"].updateLineStyleAndMarker(StyleChooser.STYLE_DASHED, StyleChooser.MARKER_X)
            self.__style_choosers["min-max_style"].updateLineStyleAndMarker(StyleChooser.STYLE_DASHED, StyleChooser.MARKER_OFF)
            self.__style_choosers["p10-p90_style"].updateLineStyleAndMarker(StyleChooser.STYLE_AREA, StyleChooser.MARKER_OFF)
            self.__style_choosers["p33-p67_style"].updateLineStyleAndMarker(StyleChooser.STYLE_AREA, StyleChooser.MARKER_OFF)


        self.blockSignals(blocked)
        self.emitChange()

