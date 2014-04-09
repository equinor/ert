from PyQt4.QtCore import Qt, SIGNAL, pyqtSlot, SLOT, pyqtSignal
from PyQt4.QtGui import QWidget, QDialog, QVBoxLayout, QLayout, QPushButton, QHBoxLayout


class ClosableDialog(QDialog):

    def __init__(self, title, widget, parent=None):
        QDialog.__init__(self, parent)

        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCancelButtonHint)


        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetFixedSize) # not resizable!!!
        layout.addWidget(widget)

        self.__button_layout = QHBoxLayout()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.__button_layout.addStretch()
        self.__button_layout.addWidget(self.close_button)

        layout.addStretch()
        layout.addLayout(self.__button_layout)

        self.setLayout(layout)


    def disableCloseButton(self):
        self.close_button.setEnabled(False)

    def enableCloseButton(self):
        self.close_button.setEnabled(True)

    def keyPressEvent(self, q_key_event):
        if not self.close_button.isEnabled() and q_key_event.key() == Qt.Key_Escape:
            pass
        else:
            QDialog.keyPressEvent(self, q_key_event)

    def addButton(self, caption, listner):
        button = QPushButton(caption)
        self.__button_layout.insertWidget(1,button)
        button.clicked.connect(listner)
