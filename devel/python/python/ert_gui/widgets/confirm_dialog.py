from PyQt4.QtCore import Qt, pyqtSignal
from PyQt4.QtGui import QDialog, QVBoxLayout, QLayout, QPushButton, QHBoxLayout, QLabel


class ConfirmDialog(QDialog):

    confirmButtonPressed = pyqtSignal()

    def __init__(self, title, text, parent=None):
        QDialog.__init__(self, parent)

        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCancelButtonHint)


        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetFixedSize) # not resizable!!!
        label = QLabel(text)
        layout.addWidget(label)

        button_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirmButtonPressed.emit)

        self.reject_button = QPushButton("Reject")
        self.reject_button.clicked.connect(self.accept)
        button_layout.addWidget(self.confirm_button)
        button_layout.addStretch()
        button_layout.addWidget(self.reject_button)

        layout.addStretch()
        layout.addLayout(button_layout)

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

