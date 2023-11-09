from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ert.analysis import SmootherSnapshot
from ert.run_models import (
    RunModelEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateEndEvent,
)


class UpdateWidget(QWidget):
    def __init__(self, iteration: int, parent=None) -> None:
        super().__init__(parent)

        self._iteration = iteration

        progress_label = QLabel("Progress:")
        self._progress_msg = QLabel()

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(0)

        widget = QWidget()
        msg_layout = QHBoxLayout()
        widget.setLayout(msg_layout)

        self._msg_list = QListWidget()
        self._msg_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

        msg_layout.addWidget(self._msg_list)

        self._tab_widget = QTabWidget()
        self._tab_widget.addTab(widget, "Status")
        self._tab_widget.setTabBarAutoHide(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(100, 20, 100, 20)

        top_layout = QHBoxLayout()
        top_layout.addWidget(progress_label)
        top_layout.addWidget(self._progress_msg)
        top_layout.addStretch()

        layout.addLayout(top_layout)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._tab_widget)

        self.setLayout(layout)

    @property
    def iteration(self) -> int:
        return self._iteration

    def _insert_report_tab(self, smoother_snapshot: SmootherSnapshot) -> None:
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        case_layout = QHBoxLayout()
        case_layout.addWidget(QLabel("Parent ensemble:"))
        case_layout.addWidget(QLabel(smoother_snapshot.source_case))
        case_layout.addSpacing(40)

        case_layout.addWidget(QLabel("Target ensemble:"))
        case_layout.addWidget(QLabel(smoother_snapshot.target_case))
        case_layout.addSpacing(40)

        case_layout.addWidget(QLabel("Alpha:"))
        case_layout.addWidget(QLabel(str(smoother_snapshot.alpha)))
        case_layout.addSpacing(40)

        case_layout.addWidget(QLabel("Global scaling:"))
        case_layout.addWidget(QLabel(str(smoother_snapshot.global_scaling)))
        case_layout.addSpacing(40)

        case_layout.addWidget(QLabel("Standard cutoff:"))
        case_layout.addWidget(QLabel(str(smoother_snapshot.std_cutoff)))

        layout.addLayout(case_layout)
        layout.addSpacing(20)

        for (
            update_step_name,
            update_step,
        ) in smoother_snapshot.update_step_snapshots.items():
            update_step_name_label = QLabel(update_step_name)

            table = QTableWidget()
            table.setColumnCount(4)
            table.setRowCount(len(update_step))
            table.setHorizontalHeaderLabels(
                ["", "Observed history", "Simulated data", "Status"]
            )
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.horizontalHeader().setStretchLastSection(True)
            table.setColumnWidth(0, 200)
            table.setColumnWidth(1, 350)
            table.setColumnWidth(2, 250)

            for nr, step in enumerate(update_step):
                obs_std = (
                    f"{step.obs_std:.3f}"
                    if step.obs_scaling == 1
                    else f"{step.obs_std * step.obs_scaling:.3f} ({step.obs_std:<.3f} * {step.obs_scaling:.3f})"
                )
                table.setItem(nr, 0, QTableWidgetItem(f"{step.obs_name:20}"))
                table.setItem(
                    nr,
                    1,
                    QTableWidgetItem(f"{step.obs_val:>16.3f} +/- {obs_std:<21}"),
                )
                table.setItem(
                    nr,
                    2,
                    QTableWidgetItem(
                        f"{step.response_mean:>21.3f} +/- {step.response_std:<16.3f}"
                    ),
                )
                table.setItem(nr, 3, QTableWidgetItem(f"{step.status.capitalize()}"))

            layout.addWidget(update_step_name_label)
            layout.addWidget(table)

        self._tab_widget.setCurrentIndex(self._tab_widget.addTab(widget, "Report"))

    @Slot()
    def begin(self) -> None:
        pass

    @Slot(RunModelUpdateEndEvent)
    def end(self, event: RunModelUpdateEndEvent) -> None:
        self._progress_msg.setText("Update completed")
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(1)
        self._progress_bar.setValue(1)

        if event.smoother_snapshot:
            self._insert_report_tab(event.smoother_snapshot)

    @Slot(RunModelEvent)
    def update_status(self, event: RunModelEvent) -> None:
        if isinstance(event, RunModelStatusEvent):
            item = QListWidgetItem()
            item.setText(event.msg)
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self._msg_list.addItem(item)
        elif isinstance(event, RunModelTimeEvent):
            self._progress_msg.setText(
                f"Estimated remaining time for current step {event.remaining_time:.2f}s"
            )
