from typing import Optional

import pandas as pd
from dateutil import parser
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget

from everest.config import EverestConfig
from everest.simulator import JOB_FAILURE, JOB_SUCCESS
from ieverest.utils import load_ui, remove_layout_item
from ieverest.widgets import BatchStatusWidget, PlotWidget
from ieverest.widgets.batch_status_widget import get_job_status


class MonitorWidget(QWidget):
    """A widget for running and monitoring an Everest configuration"""

    start_opt_requested = Signal()
    stop_opt_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        self._config: Optional[EverestConfig] = None

        super(MonitorWidget, self).__init__(parent)

        load_ui("monitor_widget.ui", self)
        self.stop_btn.setEnabled(False)

        # Insert stretch so all batches widgets are pushed at the top
        self.batches_wdg.layout().insertStretch(0, 10)

        self.start_btn.clicked.connect(self.start_opt_requested)
        self.stop_btn.clicked.connect(self.stop_opt_requested)
        self.export_btn.clicked.connect(self.export_requested)

        # Individual plots as tabs
        self.plot_widget_batch_timings = PlotWidget(parent=self)

        # Tab with combo box and plotwidget
        self.vbox_widget = QWidget(parent=self)
        vbox_layout = QVBoxLayout()

        self.objective_functions_to_render = {}
        self._current_accepted_points = []
        self.combobox_objective_funcs = QComboBox(parent=self)
        self.combobox_objective_funcs.currentIndexChanged.connect(
            lambda: self.update_objective_plot(
                str(self.combobox_objective_funcs.currentText())
            )
        )

        self.plot_widget_selected_objective = PlotWidget(parent=self)
        vbox_layout.addWidget(self.plot_widget_selected_objective)
        vbox_layout.addWidget(self.combobox_objective_funcs)
        self.vbox_widget.setLayout(vbox_layout)

        self.tabbed_plot_widget.addTab(self.vbox_widget, "Objective Func(s)")
        self.tabbed_plot_widget.addTab(self.plot_widget_batch_timings, "Batch Timings")

    @property
    def config(self) -> EverestConfig:
        assert self._config is not None
        assert isinstance(self._config, EverestConfig)
        return self._config

    def set_config(self, config: EverestConfig):
        self._config = config

    def create_objective_fn_widgets_from_config(self):
        _objective_names = [func.name for func in self.config.objective_functions]
        _objective_names.append("avg_objective")

        # Create combo box with objective names + the total average
        self.combobox_objective_funcs.clear()
        self.combobox_objective_funcs.addItems(_objective_names)
        self.vbox_widget.update()

    def _add_or_update_batch_widget(self, status):
        batch = int(status["batch_number"])
        wdg = self.batches_wdg.findChild(BatchStatusWidget, f"Batch {batch}")
        if wdg:
            wdg.status = status
        else:
            wdg = BatchStatusWidget(
                batch_name=f"Batch {batch}", parent=self.batches_wdg
            )
            wdg.status = status
            self.batches_wdg.layout().insertWidget(0, wdg)

    def reset(self):
        """Clear the content of the widget

        Call this before a new workflow is started
        """
        # Remove all the widgets from the scroll area
        while self.batches_wdg.layout().count() > 0:
            remove_layout_item(self.batches_wdg.layout(), 0)
        self.batches_wdg.layout().insertStretch(0, 10)
        self.update_batch_timings_plot()

        self.objective_functions_to_render = {}

        # Force clear plots if they had previous displayed data
        self.plot_widget_selected_objective.clear()
        self.plot_widget_batch_timings.clear()

    def update_objective_plot(self, selected_name):
        """
        Draw updated objectives plot
        :param selected_name: current selection from the objective function combo box.
        """
        if selected_name in self.objective_functions_to_render:
            self.plot_widget_selected_objective.render_line_plot(
                name=selected_name,
                values=self.objective_functions_to_render[selected_name],
                accepted_indices=self._current_accepted_points,
            )

    def update_optimization_progress(self, progress):
        """
        Parameters
        ----------
        progress: dict
        """
        if "objectives_history" not in progress:
            return
        if not progress["objectives_history"]:
            return

        self.objective_functions_to_render = progress["objectives_history"]
        self.objective_functions_to_render["avg_objective"] = progress[
            "objective_history"
        ]
        self._current_accepted_points = progress["accepted_control_indices"]

        # Update the graph
        self.update_objective_plot(str(self.combobox_objective_funcs.currentText()))

    def _get_batch_type(self, jobs):
        """Determine the type of a batch depending on number of jobs"""
        _F = "function evaluation"
        _G = "gradient_evaluation"
        _FG = "function and gradient"
        _U = "unknown"
        realizations_num = len(self.config.model.realizations)
        sim_per_eval = realizations_num
        # 5 is the default value from everest2ropt.py
        sim_per_grad = realizations_num * (
            self.config.optimization.perturbation_num or 5
        )

        if len(jobs) == sim_per_eval + sim_per_grad:
            return _FG
        if sim_per_eval == sim_per_grad:  # can't tell just by using len
            return _U
        if len(jobs) == sim_per_eval:
            return _F
        if len(jobs) == sim_per_grad:
            return _G
        return _U

    def update_batch_timings_plot(self):
        """Update the batch timings plot

        The execution time of a batch is deduced by the timing info of the
        forward models in each job in the batch
        """
        # NOTE: information about batches is stored in batches widget. And it
        # is also read out of those widgets to determine timing information for
        # the boxplot. This is arguably a bad choice, but that's where we ended
        # up via incremental development.
        # If we'll ever consider refactoring this, a possible better approach
        # can be to have the data stored in this class so that it can be
        # accessed directly for constructing the boxplots, and the batch
        # widgets should just refer to it.

        batch_widgets = self.batches_wdg.findChildren(BatchStatusWidget)
        # Don't bother if user not looking or no data to show
        if not batch_widgets:
            return

        # Create data frame from batch job timings.
        data = {"batch": [], "time": [], "batch_type": []}
        for wdg in reversed(batch_widgets):
            jobs = wdg.status["progress"]
            batch_finished = all(
                get_job_status(job) in (JOB_SUCCESS, JOB_FAILURE) for job in jobs
            )
            if not batch_finished:
                continue  # batch still running, no timing info

            batch_type = self._get_batch_type(jobs)

            for job in jobs:
                if not job:
                    continue
                start_times = [
                    fm["start_time"]
                    for fm in job
                    if fm and fm["start_time"] is not None
                ]
                start_times = [
                    parser.parse(time, fuzzy=True)
                    for time in start_times
                    if isinstance(time, str)
                ]
                end_times = [
                    fm["end_time"] for fm in job if fm and fm["end_time"] is not None
                ]
                end_times = [
                    parser.parse(time, fuzzy=True)
                    for time in end_times
                    if isinstance(time, str)
                ]
                duration = (
                    0
                    if not start_times or not end_times
                    else (max(end_times) - min(start_times)).total_seconds()
                )
                data["batch"].append(wdg.batch_name)
                data["time"].append(duration)
                data["batch_type"].append(batch_type)
        df = pd.DataFrame(data)

        # Set new data & reverse columns to order. Batch 1, Batch 2, ...
        df = df.rename(
            columns={"batch": "", "time": "Time (sec)", "batch_type": "Type"}
        )

        # Render plot
        # TODO: In order to render batch timing boxplots in different colors
        # add another column ie. 'Batch Type', with string values indicating
        # for example {'gradient', 'function', 'both'} and set the parameter
        # 'hue="Batch Type"' in the method below. Now we don't have
        # a clean way to get information about the batch type from ropt
        self.plot_widget_batch_timings.render_box_plot(
            data=df,
            x="",
            y="Time (sec)",
            hue="Type",
        )

    def update_status(self, workflow_status=None, workflow_is_running=None):
        """Update the internal status and redraw the ui."""
        if workflow_is_running is not None:
            self.start_btn.setEnabled(not workflow_is_running)
            self.stop_btn.setEnabled(workflow_is_running)
            self.export_btn.setEnabled(not workflow_is_running)
            self.repaint()

        if workflow_status is not None:
            if isinstance(workflow_status, list):
                workflow_status.sort(key=lambda k: k["batch_number"])
                for status in workflow_status:
                    self._add_or_update_batch_widget(status)
            else:
                self._add_or_update_batch_widget(workflow_status)

        self.update_batch_timings_plot()
