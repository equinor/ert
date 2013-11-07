from PyQt4.QtCore import Qt, SIGNAL, pyqtSlot, SLOT
from PyQt4.QtGui import QWidget, QDialog, QVBoxLayout, QLayout, QPushButton, QHBoxLayout


class ClosableDialog(QDialog):

    def __init__(self, title, widget, parent=None):
        QDialog.__init__(self, parent)

        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)


        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetFixedSize) # not resizable!!!
        layout.addWidget(widget)

        button_layout = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_button)

        layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)
