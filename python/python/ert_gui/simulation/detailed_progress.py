import math
import time
try:
  from PyQt4.QtCore import QTimer, pyqtSignal, QVariant, Qt, QAbstractTableModel
  from PyQt4.QtGui import QWidget, QPainter, QColor, QFrame, QGridLayout, QImage, QDialog, QTableView, QLabel, QPen
except ImportError:
  from PyQt5.QtCore import QTimer, pyqtSignal, QVariant, Qt, QAbstractTableModel
  from PyQt5.QtWidgets import QWidget, QFrame, QDialog, QTableView, QLabel, QGridLayout
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

        self.setMinimumHeight(200)

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
        nr_realizations = max([iens for iens, progress in self._current_progress]) + 1
        fm_size = max([len(progress) for iens, progress in self._current_progress])
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

    def data(self, index, role):
        if not index.isValid():
            return QVariant()

        status = self.model_data[index.row()][self.get_column_index("status")]
        if role == Qt.BackgroundColorRole:
            color = QColor(*self.state_colors[status])
            color.setAlpha(color.alpha()/2)
            return QVariant(color)

        if role != Qt.DisplayRole:
            return QVariant()

        if self.get_column_name(index.column()).find("time") >= 0:
            if status == "Pending" or status == "Waiting":
                return QVariant()

            timestamp = eval(self.model_data[index.row()][index.column()])
            return QVariant(time.ctime(timestamp))

        return QVariant(self.model_data[index.row()][index.column()])

    def headerData(self, index, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            return QVariant(self.get_column_name(index))
        if orientation == Qt.Vertical:
            return QVariant(index)


class DetailedProgressDialog(QDialog):
    def __init__(self, parent, state_colors):
        super(DetailedProgressDialog, self).__init__(parent)
        self.setWindowTitle("Realization Progress")
        layout = QGridLayout(self)

        self.detailed_progress_widget = DetailedProgress(state_colors, self)
        self.overview_label = QLabel("Realizations")

        self.single_view = QTableView()
        self.single_view.setModel(SingleProgressModel(self.single_view,state_colors))
        self.single_view_label = QLabel("Realization details")

        self.detailed_progress_widget.clicked.connect(self.show_selection)

        layout.addWidget(self.single_view_label, 2, 0)
        layout.addWidget(self.overview_label, 0, 0)
        layout.addWidget(self.single_view, 3, 0)
        layout.addWidget(self.detailed_progress_widget, 1, 0)

        self.detailed_progress_widget.show()
        self.setLayout(layout)

        self.layout().setRowStretch(1,2)
        self.layout().setRowStretch(3, 1)
        self.progress = None
        self.selected_realization = None
        self.resize(parent.width(), parent.height())

    def set_progress(self, progress, iteration):
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

        model_data = []
        jobs = self.progress[self.selected_realization]
        headers = []
        for job in jobs:
            data = job.dump_data()
            row = [str(data[key]) for key in data]
            model_data.append(row)
            headers = list(data.keys())

        self.single_view.model().update_data(headers, model_data)
        self.single_view.resizeColumnsToContents()
        self.single_view.model().modelReset.emit()

