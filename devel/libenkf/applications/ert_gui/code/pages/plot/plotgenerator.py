import os
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from PyQt4.QtGui import QFrame, QSizePolicy, QVBoxLayout, QCursor, QDialog, QDialogButtonBox
from pages.plot.plotfigure import PlotFigure
from PyQt4.QtCore import Qt, QPoint, QSize, SIGNAL
from PyQt4.QtGui import QProgressBar, QApplication
import threading
from plotsettingsxml import PlotSettingsLoader
from plotsettings import PlotSettings

class PlotGenerator(QFrame):

    def __init__(self, plot_path, plot_config_path):
        QFrame.__init__(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.plot_figure = PlotFigure()
        self.canvas = FigureCanvas(self.plot_figure.getFigure())
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        size = QSize(297*2, 210*2) # A4 aspectratio
        self.canvas.setMaximumSize(size)
        self.canvas.setMinimumSize(size)
        self.setMaximumSize(size)
        self.setMinimumSize(size)

        self.popup = Popup(self)
        self.plot_config_loader = PlotSettingsLoader()
        self.plot_settings = PlotSettings()
        self.plot_settings.setPlotPath(plot_path)
        self.plot_settings.setPlotConfigPath(plot_config_path)
        self.connect(self.popup, SIGNAL('updateProgress(int)'), self.updateProgress)

    def updateProgress(self, progress = 1):
        value = self.popup.progress_bar.value()
        self.popup.progress_bar.setValue(value + progress)

    def save(self, plot_data):
        name = plot_data.getSaveName()
        print name
        self.plot_config_loader.load(name, self.plot_settings)

        self.startSaveThread(plot_data, self.plot_settings)
        self.popup.exec_()
        

    def startSaveThread(self, plot_data, plot_settings):
        self.runthread = threading.Thread(name="plot_saving")

        def run():
            self.plot_figure.drawPlot(plot_data, plot_settings)
            self.canvas.draw()

            QApplication.processEvents()

            plot_path = plot_settings.getPlotPath()
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

            path = plot_path + "/" + plot_data.getTitle()
            self.plot_figure.getFigure().savefig(path + ".png", dpi=400, format="png")

            QApplication.processEvents()

            self.plot_figure.getFigure().savefig(path + ".pdf", dpi=400, format="pdf")

            self.popup.emit(SIGNAL('updateProgress(int)'), 1)

            QApplication.processEvents()
            
            self.popup.ok_button.setEnabled(True)

        self.runthread.run = run
        self.runthread.start()


class Popup(QDialog):
    def __init__(self, widget, parent = None):
        QDialog.__init__(self, parent)
        self.setModal(True)
        self.setWindowTitle("Plot save progress")

        layout = QVBoxLayout()
        layout.addWidget(widget)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok, Qt.Horizontal, self)
        layout.addWidget(buttons)
        
        self.setLayout(layout)

        self.connect(buttons, SIGNAL('accepted()'), self.accept)

        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)

    def closeEvent(self, event):
        """Ignore clicking of the x in the top right corner"""
        event.ignore()

    def keyPressEvent(self, event):
        """Ignore ESC keystrokes"""
        if not event.key() == Qt.Key_Escape:
            QDialog.keyPressEvent(self, event)

