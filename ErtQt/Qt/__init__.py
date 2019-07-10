try:
    from PyQt5.Qt import *

except ImportError:
    from PyQt4.Qt import *

    # Monkey patch QFileDialog to conform to Qt5 API
    QFileDialog.getOpenFileName = QFileDialog.getOpenFileNameAndFilter
    QFileDialog.getOpenFileNames = QFileDialog.getOpenFileNamesAndFilter
    QFileDialog.getSaveFileName = QFileDialog.getSaveFileNameAndFilter


