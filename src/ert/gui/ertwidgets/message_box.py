from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QStyle,
    QTextEdit,
)


# This might seem like NIH, however the QtMessageBox is notoriously
# hard to work with if you need anything like for example resizing.
class ErtMessageBox(QDialog):
    def __init__(self, text, detailed_text, parent=None):
        super().__init__(parent)
        box = QDialogButtonBox(
            QDialogButtonBox.Ok,
            centerButtons=True,
        )
        box.accepted.connect(self.accept)

        self.details_text = QTextEdit()
        self.details_text.setPlainText(detailed_text)
        self.details_text.setReadOnly(True)

        self.label_text = QLabel(text)
        self.label_icon = QLabel()
        icon = self.style().standardIcon(QStyle.SP_MessageBoxCritical)
        self.label_icon.setPixmap(icon.pixmap(32))

        lay = QGridLayout(self)
        lay.addWidget(self.label_icon, 1, 1, 1, 1)
        lay.addWidget(self.label_text, 1, 2, 1, 2)
        lay.addWidget(self.details_text, 2, 1, 1, 3)
        lay.addWidget(box, 3, 1, 1, 3)

        self.setMinimumSize(640, 100)
