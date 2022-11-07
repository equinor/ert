import sys
import traceback

from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QAction, QVBoxLayout, QWidget

from ert.gui.ertwidgets import resourceIcon


class CustomNavigationToolbar(NavigationToolbar2QT):
    customizationTriggered = Signal()

    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)

        gear = resourceIcon("edit.svg")
        customize_action = QAction(gear, "Customize", self)
        customize_action.setToolTip("Customize plot settings")
        customize_action.triggered.connect(self.customizationTriggered)

        for action in self.actions():
            if str(action.text()).lower() == "subplots":
                self.removeAction(action)

            if str(action.text()).lower() == "customize":
                self.insertAction(action, customize_action)
                self.removeAction(action)
                break


class PlotWidget(QWidget):
    customizationTriggered = Signal()

    def __init__(self, name, plotter, parent=None):
        QWidget.__init__(self, parent)

        self._name = name
        self._plotter = plotter
        """:type: list of functions """

        self._figure = Figure()
        self._figure.set_tight_layout(True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setParent(self)
        self._canvas.setFocusPolicy(Qt.StrongFocus)
        self._canvas.setFocus()

        vbox = QVBoxLayout()
        vbox.addWidget(self._canvas)
        self._toolbar = CustomNavigationToolbar(self._canvas, self)
        self._toolbar.customizationTriggered.connect(self.customizationTriggered)
        vbox.addWidget(self._toolbar)
        self.setLayout(vbox)

        self._dirty = True
        self._active = False
        self.resetPlot()

    def resetPlot(self):
        self._figure.clear()

    @property
    def name(self) -> str:
        return self._name

    def updatePlot(self, plot_context, case_to_data_map, observations):
        self.resetPlot()
        try:
            self._plotter.plot(
                self._figure, plot_context, case_to_data_map, observations
            )
            self._canvas.draw()
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            sys.stderr.write("-" * 80 + "\n")
            traceback.print_tb(exc_tb)
            sys.stderr.write(f"Exception type: {exc_type.__name__}\n")
            sys.stderr.write(f"{e}\n")
            sys.stderr.write("-" * 80 + "\n")
            sys.stderr.write(
                "An error occurred during plotting. "
                "This stack trace is helpful for diagnosing the problem."
            )
