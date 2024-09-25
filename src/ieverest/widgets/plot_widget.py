# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from qtpy.QtWidgets import QVBoxLayout, QWidget

sns.set_style("whitegrid")


class PlotWidget(QWidget):
    """
    Usage:
    - Set widget into UI (ie. ieverest.utils.replace_widget_by_name)
    """

    def __init__(self, parent=None):
        """
        Initialize UI objects
        """
        super(PlotWidget, self).__init__(parent)
        self.fig, self._axes = plt.subplots()
        self._canvas = FigureCanvas(self.fig)
        self._canvas.setParent(self)
        self._toolbar = NavigationToolbar(self._canvas, parent=parent or self)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._toolbar)
        self.layout().addWidget(self._canvas)

    def render_box_plot(self, data, **kwargs):
        if data.shape[0] > 0:
            self._axes.clear()
            sns.boxplot(data=data, ax=self._axes, **kwargs)

        self._canvas.draw()

    def render_line_plot(self, name, values, accepted_indices=None):
        """
        Display a named data series
        :param name: data series name
        :param values: data series values
        :param accepted_indices: mark the data series values that will be
        rendered as part of the line plot. Values corresponding to indices not
        part of the accepted_indices will be displayed as scattered points in
        the plot.
        """
        if accepted_indices is None:
            accepted_indices = range(len(values))

        if len(values) > 0:
            accepted = {"index": [], "values": []}
            not_accepted = {"index": [], "values": []}
            for i, val in enumerate(values):
                if i in accepted_indices:
                    accepted["index"].append(i)
                    accepted["values"].append(val)
                else:
                    not_accepted["index"].append(i)
                    not_accepted["values"].append(val)

            self._axes.clear()
            self._axes.plot(accepted["index"], accepted["values"], "o-", label=name)
            self._axes.plot(
                not_accepted["index"],
                not_accepted["values"],
                "ko",
                label=name + " rejected",
            )

            self._axes.legend()
            self._canvas.draw()
        else:
            self.clear()

    def clear(self):
        """
        Call when the plot widget needs to be cleared of previous displayed data.
        """
        self._axes.clear()
        self._canvas.draw()
