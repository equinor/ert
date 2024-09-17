from typing import ClassVar

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QIcon, QPixmap
from qtpy.QtWidgets import QLabel, QSizePolicy, QSpacerItem, QStyle, QWidget, qApp

from everest.simulator import JOB_FAILURE, JOB_RUNNING, JOB_SUCCESS, JOB_WAITING
from ieverest.utils import load_ui


def get_job_status(job_info):
    """Return the job status depending on the forward models.

    job_info is supposed to be a list of dictionaries, where each
    dictionary represent a forward model in a job. And each of the
    dictionary should have a 'status' key.
    """
    #       If the list is empty, it means we have no information about
    #       the job. This is most likely because the job has not started
    #       yet, so we identify it as "Waiting"
    if len(job_info) == 0:
        return JOB_WAITING

    states = {fm.get("status") for fm in job_info}

    # at least one failure -> failure
    if JOB_FAILURE in states:
        return JOB_FAILURE

    # no failure and at least one running -> running
    if JOB_RUNNING in states:
        return JOB_RUNNING

    # no failure nor running, at least one not recognized -> unknown (None)
    if len(states - {JOB_FAILURE, JOB_RUNNING, JOB_SUCCESS, JOB_WAITING}) > 0:
        return None

    # no fail/run/unknown, at least one waiting -> waiting
    if JOB_WAITING in states:
        return JOB_WAITING

    return JOB_SUCCESS


class BatchStatusWidget(QWidget):
    """A widget monitoring the status of a forward model batch"""

    _ICON_SIZE = QSize(24, 24)
    _STATUS_ICONS: ClassVar = {
        JOB_SUCCESS: "emblem-default",
        JOB_WAITING: "emblem-new",
        JOB_RUNNING: "emblem-system",
        JOB_FAILURE: "emblem-important",
    }
    _DEFAULT_STATUS_ICON = "dialog-question"
    _STATUS_TEXT: ClassVar = {
        JOB_SUCCESS: "v",
        JOB_WAITING: "-",
        JOB_RUNNING: "*",
        JOB_FAILURE: "x",
    }
    _DEFAULT_STATUS_TEXT = "?"

    def __init__(self, batch_name="", status=None, parent=None):
        super(BatchStatusWidget, self).__init__(parent)

        # This needed to be moved before the load_ui call below, a bit unclear
        # why, but apparently some signal gets fired during load_ui, which
        # accesses status.
        self._status = None

        load_ui("batch_status_widget.ui", self)

        self._jobs_wdgs = []
        self._spacer = QSpacerItem(0, 0, hPolicy=QSizePolicy.Expanding)
        self._icons_per_row = -1
        self._icons_num = -1
        self._latest_progress = None
        self.jobs_wdg.layout().addItem(self._spacer, 0, 0)

        self.setObjectName(batch_name)
        self.batch_name = batch_name
        self.status = status

    def paintEvent(self, event):
        # Need to update the ui whenever a paintEvent occurs (so that we get
        # the correct number of icons per row)
        super(BatchStatusWidget, self).paintEvent(event)
        self.update_ui()

    @property
    def batch_name(self):
        return str(self.batch_name_lbl.text())

    @batch_name.setter
    def batch_name(self, t):
        self.batch_name_lbl.setText(t)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, s):
        self._status = s
        self.update_ui()

    def update_ui(self):
        if self._status is None:
            self.setEnabled(False)
            return
        status = self._status.get("status")
        progress = self._status.get("progress")
        if None in (status, progress):
            self.setEnabled(False)
            return
        self.setEnabled(True)
        self._set_status(status)
        self._set_progress(progress)

    def _set_status(self, s):
        """Update the overview section according to the given data"""

        def _txt(val, color):
            if val <= 0:
                return str(val)
            fmt = '<span style="font-weight:{}; color:{};">{}</span>'
            return fmt.format(600, color, val)

        waiting = s.waiting + s.pending
        self.waiting_lbl.setText(_txt(waiting, "#ff7f00"))
        self.running_lbl.setText(_txt(s.running, "#377eb8"))
        self.finished_lbl.setText(_txt(s.complete, "#4daf4a"))
        self.failed_lbl.setText(_txt(s.failed, "#e41a1c"))

    def _set_progress(self, progress):
        """Update the details section according to the given data"""

        # Determines the number of icons per row depending on the width of the
        # current visible area of the widget (and allow for a scrollbar)
        tw = (
            self.jobs_wdg.visibleRegion().boundingRect().width()
            - qApp.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        )
        iw = self._ICON_SIZE.width()
        ih = self._ICON_SIZE.height()
        hs = self.jobs_wdg.layout().horizontalSpacing()
        vs = self.jobs_wdg.layout().verticalSpacing()
        icons_per_row = (tw + hs) / (iw + hs)

        # If nothing has changed do not update the UI
        # Some of the calls further down trigger a paintEvent on the parent
        # widget. If the update is always done, regardless of the update,
        # we get into an infinite loop of paint events
        if (
            self._latest_progress == progress
            and self._icons_num == len(self._jobs_wdgs)
            and self._icons_per_row == icons_per_row
        ):
            return

        self._latest_progress = progress
        self._icons_num = len(self._jobs_wdgs)
        self._icons_per_row = icons_per_row

        # create new job buttons if necessary
        while len(self._jobs_wdgs) < len(progress):
            wdg = QLabel(self.jobs_wdg)
            self._jobs_wdgs.append(wdg)
            wdg.setFixedSize(self._ICON_SIZE)

        # update the buttons
        for wdg, info in zip(self._jobs_wdgs, progress):
            job_status = get_job_status(info)
            icon_name = self._STATUS_ICONS.get(job_status, self._DEFAULT_STATUS_ICON)

            if QIcon.hasThemeIcon(icon_name):
                pixmap = QIcon.fromTheme(icon_name).pixmap(self._ICON_SIZE)
                wdg.setPixmap(pixmap)
                wdg.setText("")
            else:
                text = self._STATUS_TEXT.get(job_status, self._DEFAULT_STATUS_TEXT)
                wdg.setPixmap(QPixmap())
                wdg.setText(text)

            wdg.setToolTip(
                "Status: {status}\n{job_info}".format(
                    status=job_status, job_info=self._fm_progress_message(info)
                )
            )

        # Remove items from layout
        for wdg in self._jobs_wdgs:
            self.jobs_wdg.layout().removeWidget(wdg)
        self.jobs_wdg.layout().removeItem(self._spacer)

        row = 0
        col = 0
        for wdg in self._jobs_wdgs:
            if col == icons_per_row:
                col = 0
                row += 1
            self.jobs_wdg.layout().addWidget(wdg, row, col, Qt.AlignTop)
            col += 1
        self.jobs_wdg.setMinimumHeight(ih * (row + 1) + vs * row)

        if len(self._jobs_wdgs) < icons_per_row:
            # Add a spacer at the end of row 0 to align widgets on the left side
            self.jobs_wdg.layout().addItem(self._spacer, 0, col + 1)

    @staticmethod
    def _fm_progress_message(fm_info):
        if len(fm_info) == 0:
            return "No detailed information yet"
        first_job = next(iter(fm_info), {})

        entries = []
        for job in fm_info:
            job_name = job.get("name", "unknown")
            job_status = job.get("status", "unknown")
            job_start = job.get("start_time", "unknown")
            job_end = job.get("end_time", "unknown")
            base_msg = "- {}: {}".format(job_name, job_status)

            if job_status == JOB_WAITING:
                entries.append(base_msg)
                continue

            msg_parts = [base_msg, " (start: {}".format(job_start)]
            if job_status == JOB_RUNNING:
                msg_parts.append(")")
            else:
                msg_parts.append(", end: {})".format(job_end))
            if job.get("error"):
                msg_parts.append("\n    {}".format(job["error"]))
            entries.append("".join(msg_parts))
        return (
            "Realization: {realization}"
            "\nSimulation: {simulation}"
            "\nDetails:\n {entries}".format(
                realization=first_job.get("realization", "unknown"),
                simulation=first_job.get("simulation", "unknown"),
                entries="\n".join(entries),
            )
        )
