import math
import time

from qtpy.QtCore import Signal, Qt, QAbstractTableModel
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableView,
    QTextEdit,
    QWidget
)
from qtpy.QtGui import QPainter, QColor, QImage, QPen

from res.job_queue import JobStatusType

from ert_gui.tools.file import FileDialog


class DetailedProgress(QFrame):
    clicked = Signal(int)

    def __init__(self, state_colors, parent):
        super(DetailedProgress, self).__init__(parent)
        self.setLineWidth(1)

        self.state_colors = state_colors
        self._current_iteration = 0
        self._current_progress = []
        self.selected_realization = -1
        self.grid_height = -1
        self.grid_width = -1

    def mousePressEvent(self, event):
        super(DetailedProgress, self).mousePressEvent(event)
        position = event.pos()

        x = int((float(position.x()) / self.width()) * self.grid_width)
        y = int((float(position.y()) / self.height()) * self.grid_height)
        index = y * self.grid_width + x

        self.selected_realization = index
        self.clicked.emit(index)
        self.update()

    def draw_window(self, x, y, progress, render_image):
        nr_jobs = len(progress)
        grid_size = int(math.ceil(math.sqrt(nr_jobs)))

        for index, job in enumerate(progress.values() if isinstance(progress, dict) else progress):
            y_off = int(index / grid_size)
            x_off = index - (y_off * grid_size)
            color = QColor(*self.state_colors[job.status])
            render_image.setPixel(x + x_off, y + y_off, color.rgb())

    def set_progress(self, progress, iteration):
        self.setMinimumHeight(200)
        self._current_progress = sorted([(iens, jobs, status) for iens, (jobs, status) in progress.items()],
                                        key=lambda x: x[0])
        self._current_iteration = iteration
        self.update()

    def has_realization_failed(self, progress):
        for job in progress.values() if isinstance(progress, dict) else progress:
            if job.status == 'Failure':
                return True

        return False

    def paintEvent(self, event):
        super(DetailedProgress, self).paintEvent(event)
        if not self._current_progress:
            return

        painter = QPainter(self)
        width = self.width()
        height = self.height()
        aspect_ratio = float(width) / height
        nr_realizations = max([iens for iens, _, _ in self._current_progress]) + 1
        fm_size = max([len(progress) for _, progress, _ in self._current_progress])
        self.grid_height = math.ceil(math.sqrt(nr_realizations / aspect_ratio))
        self.grid_width = math.ceil(self.grid_height * aspect_ratio)
        sub_grid_size = math.ceil(math.sqrt(fm_size))
        cell_height = height / self.grid_height
        cell_width = width / self.grid_width

        foreground_image = QImage(self.grid_width * sub_grid_size, self.grid_height * sub_grid_size,
                                  QImage.Format_ARGB32)
        foreground_image.fill(QColor(0, 0, 0, 0))

        for index, (iens, progress, _) in enumerate(self._current_progress):
            y = int(iens / self.grid_width)
            x = int(iens - (y * self.grid_width))
            self.draw_window(x * sub_grid_size, y * sub_grid_size, progress, foreground_image)
            painter.drawImage(self.contentsRect(), foreground_image)

        for index, (iens, progress, state) in enumerate(self._current_progress):
            y = int(iens / self.grid_width)
            x = int(iens - (y * self.grid_width))

            painter.setPen(QColor(80, 80, 80))
            painter.drawText(int(x * cell_width), int(y * cell_height), int(cell_width), int(cell_height),
                             int(Qt.AlignHCenter | Qt.AlignVCenter), str(iens))

            if iens == self.selected_realization:
                pen = QPen(QColor(240, 240, 240))
            elif (self.has_realization_failed(progress)):
                pen = QPen(QColor(*self.state_colors['Failure']))
            elif (state == JobStatusType.JOB_QUEUE_RUNNING):
                pen = QPen(QColor(*self.state_colors['Running']))
            else:
                pen = QPen(QColor(80, 80, 80))

            thickness = 4
            pen.setWidth(thickness)
            painter.setPen(pen)
            painter.drawRect(int(x * cell_width) + (thickness // 2),
                             int(y * cell_height) + (thickness // 2),
                             int(cell_width) - (thickness - 1),
                             int(cell_height) - (thickness - 1))


class SingleProgressModel(QAbstractTableModel):
    def __init__(self, parent_view, state_colors):
        super(SingleProgressModel, self).__init__(parent_view)
        self.model_data = []
        self.model_header = []
        self.state_colors = state_colors

    def update_data(self, header, data):
        self.model_data = data
        self.model_header = header

    def columnCount(self, parent=None):
        if not self.model_data:
            return 0
        return len(self.model_data[0])

    def rowCount(self, parent=None):
        return len(self.model_data)

    def get_column_name(self, index):
        if not self.model_header:
            return ""
        return self.model_header[index]

    def get_column_index(self, name):
        return self.model_header.index(name)

    def get_file_name(self, index):
        col = self.get_column_name(index.column())
        if col == 'stdout' or col == 'stderr':
            return self.model_data[index.row()][index.column()]
        return ''

    @staticmethod
    def _get_byte_with_unit(byte_count):
        suffixes = ["B", "kB", "MB", "GB", "TB", "PB"]
        power = float(10 ** 3)

        i = 0
        while byte_count >= power and i < len(suffixes) - 1:
            byte_count = byte_count / power
            i += 1

        return "{byte_count:.2f} {suffix}".format(byte_count=byte_count, suffix=suffixes[i])

    def data(self, index, role):
        if not index.isValid():
            return

        status = self.model_data[index.row()][self.get_column_index("status")]
        col = self.get_column_name(index.column())

        if role == Qt.BackgroundColorRole:
            color = QColor(*self.state_colors[status])
            color.setAlpha(round(color.alpha() / 2))
            if col == 'stdout' or col == 'stderr':
                color = QColor(100,100,100,100) # make items stand out
            return color

        if role != Qt.DisplayRole:
            return

        if self.get_column_name(index.column()).find("time") >= 0:
            if status == "Pending" or status == "Waiting":
                return

            timestamp = eval(self.model_data[index.row()][index.column()])
            return time.ctime(timestamp)

        if col == 'stdout' or col == 'stderr':
            return "OPEN"

        if col == 'current_memory_usage' or col == 'max_memory_usage':
            try:
                memory_usage = int(self.model_data[index.row()][index.column()])
            except ValueError:
                return self.model_data[index.row()][index.column()]
            else:
                return SingleProgressModel._get_byte_with_unit(memory_usage)

        return self.model_data[index.row()][index.column()]

    def headerData(self, index, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return
        if orientation == Qt.Horizontal:
            return self.get_column_name(index)
        if orientation == Qt.Vertical:
            return index


class SingleTableView(QTableView):
    def __init__(self):
        super(SingleTableView, self).__init__()
        self.open_files = {}
        self.realization = -1
        self.iteration = -1

    def mousePressEvent(self, mouse_event):
        pos = mouse_event.pos()
        index = self.indexAt(pos)
        selected_file = self.model().get_file_name(index)
        if selected_file and not selected_file in self.open_files:
            job_name = self.model().model_data[index.row()][self.model().get_column_index("name")]
            viewer = FileDialog(selected_file, job_name, index.row(), self.realization, self.iteration, self)
            self.open_files[selected_file] = viewer
            viewer.finished.connect(lambda _, f=selected_file: self.open_files.pop(f))

        elif selected_file in self.open_files:
            self.open_files[selected_file].raise_()

    def update_data(self, jobs, iteration, realization):
        self.setMinimumHeight(200)
        self.iteration = iteration
        self.realization = realization
        model_data = []
        headers = []
        for job in jobs.values() if isinstance(jobs, dict) else jobs:
            data = job.dump_data()
            row = [str(data[key]) for key in data]
            model_data.append(row)
            headers = list(data.keys())

        self.model().update_data(headers, model_data)
        self.resizeColumnsToContents()
        self.model().modelReset.emit()


class DetailedProgressWidget(QWidget):
    def __init__(self, parent, state_colors):
        super(DetailedProgressWidget, self).__init__(parent)
        self.setWindowTitle("Realization Progress")
        layout = QGridLayout(self)

        self.iterations = QTabWidget()
        self.state_colors = state_colors

        self.single_view = SingleTableView()
        self.single_view.setModel(SingleProgressModel(self.single_view, state_colors))
        self.single_view_label = QLabel("Realization details")

        layout.addWidget(self.iterations, 1, 0)
        layout.addWidget(self.single_view_label, 2, 0)
        layout.addWidget(self.single_view, 3, 0)

        self.setLayout(layout)

        self.layout().setRowStretch(1, 1)
        self.layout().setRowStretch(3, 1)
        self.progress = None
        self.selected_realization = -1
        self.current_iteration = -1
        self.resize(parent.width(), parent.height())
        self.progress = {}
        self._iter_to_tab = {}

    def set_progress(self, progress, iteration):
        if iteration < 0:
            return

        self.progress = progress
        for i in progress:  # create all detailed views if they havent been constructed yet
            if i in self._iter_to_tab:
                continue
            detailed_progress_widget = DetailedProgress(self.state_colors, self)
            detailed_progress_widget.clicked.connect(self.show_selection)
            detailed_progress_widget.set_progress(progress[i], i)
            detailed_progress_widget.show()
            self._iter_to_tab[i] = self.iterations.addTab(detailed_progress_widget,
                                                          "Realizations for iteration {}".format(i))

        current_progress_widget = self.iterations.widget(self._iter_to_tab[iteration])
        current_progress_widget.set_progress(progress[iteration], iteration)

        self.update_single_view()
        self.update()

    def show_selection(self, iens):
        if not self.progress:
            return

        self.current_iteration = self.iterations.currentIndex()

        for i in range(0, self.iterations.count()):
            if i == self.current_iteration:
                continue
            self.iterations.widget(i).selected_realization = -1

        self.selected_realization = iens
        self.single_view_label.setText("Realization id: {} in iteration {}".format(iens, self.current_iteration))
        self.update_single_view()

    def update_single_view(self):
        if not self.single_view.isVisible():
            return
        if not self.current_iteration in self.progress:
            return
        if not self.selected_realization in self.progress[self.current_iteration]:
            return

        self.single_view.update_data(self.progress[self.current_iteration][self.selected_realization][0],
                                     self.current_iteration,
                                     self.selected_realization)
