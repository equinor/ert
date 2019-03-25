import math
import time
try:
  from PyQt4.QtCore import QTimer, pyqtSignal, QVariant, Qt, QAbstractTableModel
  from PyQt4.QtGui import (QWidget,
                           QPainter,
                           QColor,
                           QFrame,
                           QGridLayout,
                           QImage,
                           QDialog,
                           QTableView,
                           QLabel,
                           QPen,
                           QPushButton,
                           QTextEdit)

except ImportError:
  from PyQt5.QtCore import QTimer, pyqtSignal, QVariant, Qt, QAbstractTableModel
  from PyQt5.QtWidgets import QWidget, QFrame, QDialog, QTableView, QLabel, QGridLayout, QPushButton, QTextEdit
  from PyQt5.QtGui import QPainter, QColor, QImage, QPen


class DetailedProgress(QFrame):
    clicked = pyqtSignal(int)

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

        for index, job in enumerate(progress):
            y_off = index / grid_size
            x_off = index - (y_off * grid_size)
            color = QColor(*self.state_colors[job.status])
            render_image.setPixel(x + x_off, y + y_off, color.rgb())

    def set_progress(self, progress, iteration):
        self.setMinimumHeight(200)
        self._current_progress = sorted(progress.items(), key=lambda x: x[0])
        self._current_iteration = iteration
        self.update()

    def has_realization_failed(self, progress):
        for job in progress:
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
        aspect_ratio = float(width)/height
        nr_realizations = max([iens for iens, _ in self._current_progress]) + 1
        fm_size = max([len(progress) for _ , progress in self._current_progress])
        self.grid_height = math.ceil(math.sqrt(nr_realizations / aspect_ratio))
        self.grid_width = math.ceil(self.grid_height * aspect_ratio)
        sub_grid_size = math.ceil(math.sqrt(fm_size))
        cell_height = height / self.grid_height
        cell_width = width / self.grid_width

        foreground_image = QImage(self.grid_width*sub_grid_size, self.grid_height*sub_grid_size, QImage.Format_ARGB32)
        foreground_image.fill(QColor(0, 0, 0, 0))

        for index, (iens, progress) in enumerate(self._current_progress):
            y = int(iens / self.grid_width)
            x = int(iens - (y * self.grid_width))
            self.draw_window(x * sub_grid_size, y * sub_grid_size, progress, foreground_image)
        painter.drawImage(self.contentsRect(), foreground_image)

        for index, (iens, progress) in enumerate(self._current_progress):
            y = int(iens / self.grid_width)
            x = int(iens - (y * self.grid_width))

            painter.setPen(QColor(80, 80, 80))
            painter.drawText(x * cell_width, y * cell_height, cell_width, cell_height, Qt.AlignHCenter | Qt.AlignVCenter, str(iens))

            if iens == self.selected_realization:
                pen = QPen(QColor(240, 240, 240))
            elif(self.has_realization_failed(progress)):
                pen = QPen(QColor(*self.state_colors['Failure']))
            else:
                pen = QPen(QColor(80, 80, 80))

            thickness = 4
            pen.setWidth(thickness)
            painter.setPen(pen)
            painter.drawRect((x * cell_width)+(thickness / 2),
                             (y * cell_height) + (thickness / 2),
                             cell_width - (thickness - 1),
                             cell_height - (thickness - 1))


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
            return self.model_data[index.row()][index.column()] + "." + str(index.row())
        return ''

    def data(self, index, role):
        if not index.isValid():
            return QVariant()

        status = self.model_data[index.row()][self.get_column_index("status")]
        col = self.get_column_name(index.column())

        if role == Qt.BackgroundColorRole:
            color = QColor(*self.state_colors[status])
            color.setAlpha(color.alpha()/2)
            if col == 'stdout' or col == 'stderr':
                color = QColor(100,100,100,100) # make items stand out
            return QVariant(color)

        if role != Qt.DisplayRole:
            return QVariant()

        if self.get_column_name(index.column()).find("time") >= 0:
            if status == "Pending" or status == "Waiting":
                return QVariant()

            timestamp = eval(self.model_data[index.row()][index.column()])
            return QVariant(time.ctime(timestamp))

        if col == 'stdout' or col == 'stderr':
            return QVariant("OPEN")

        return QVariant(self.model_data[index.row()][index.column()])

    def headerData(self, index, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            return QVariant(self.get_column_name(index))
        if orientation == Qt.Vertical:
            return QVariant(index)

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
            viewer = FileViewer(self, selected_file, job_name, index.row(), self.realization, self.iteration)
            self.open_files[selected_file] = viewer

        elif selected_file in self.open_files:
            self.open_files[selected_file].reload(selected_file)

    def update_data(self, jobs, iteration, realization):
        self.setMinimumHeight(200)
        self.iteration = iteration
        self.realization = realization
        model_data = []
        headers = []
        for job in jobs:
            data = job.dump_data()
            row = [str(data[key]) for key in data]
            model_data.append(row)
            headers = list(data.keys())

        self.model().update_data(headers, model_data)
        self.resizeColumnsToContents()
        self.model().modelReset.emit()

        for file_name in self.open_files:
            if self.open_files[file_name].isVisible():
                self.open_files[file_name].reload(file_name)


class FileViewer(QDialog):
    def __init__(self, parent, file_name, job_name, job_number, iteration, realization):
        super(FileViewer, self).__init__(parent)

        self.setWindowTitle("{} # {} Realization: {} Iteration: {}" \
                            .format(job_name, job_number, realization, iteration))

        self.text_cont = QTextEdit()
        self.text_cont.setReadOnly(True)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)

        layout = QGridLayout(self)
        layout.addWidget(self.text_cont)
        layout.addWidget(close_button)

        self.setMinimumWidth(400)
        self.setMinimumHeight(200)

        self.reload(file_name)

    def reload(self, file_name):
        with open(file_name) as f:
            text = f.read()
        self.text_cont.setText(text)
        self.show()


class DetailedProgressDialog(QDialog):
    def __init__(self, parent, state_colors):
        super(DetailedProgressDialog, self).__init__(parent)
        self.setWindowTitle("Realization Progress")
        layout = QGridLayout(self)

        self.detailed_progress_widget = DetailedProgress(state_colors, self)
        self.overview_label = QLabel("Realizations")

        self.single_view = SingleTableView()
        self.single_view.setModel(SingleProgressModel(self.single_view,state_colors))
        self.single_view_label = QLabel("Realization details")

        self.detailed_progress_widget.clicked.connect(self.show_selection)

        layout.addWidget(self.single_view_label, 2, 0)
        layout.addWidget(self.overview_label, 0, 0)
        layout.addWidget(self.single_view, 3, 0)
        layout.addWidget(self.detailed_progress_widget, 1, 0)

        self.detailed_progress_widget.show()
        self.setLayout(layout)

        self.layout().setRowStretch(1, 1)
        self.layout().setRowStretch(3, 1)
        self.progress = None
        self.selected_realization = -1
        self.current_iteration = -1
        self.resize(parent.width(), parent.height())

    def set_progress(self, progress, iteration):
        self.current_iteration = iteration
        self.progress = progress
        self.detailed_progress_widget.set_progress(progress, iteration)
        self.overview_label.setText("Realizations for iteration {}".format(iteration))
        self.update_single_view()
        self.update()

    def show_selection(self, iens):
        if not self.progress:
            return

        self.selected_realization = iens
        self.single_view_label.setText("Realization id: {}".format(iens))
        self.update_single_view()

    def update_single_view(self):
        if not self.single_view.isVisible() or not self.selected_realization in self.progress:
            return

        self.single_view.update_data(self.progress[self.selected_realization],
                                     self.current_iteration,
                                     self.selected_realization)
