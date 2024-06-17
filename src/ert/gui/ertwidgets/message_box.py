from typing import Optional

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QStyle,
    QTextEdit,
    QWidget,
)

from ert.gui.tools.file.file_dialog import (
    calculate_font_based_width,
    calculate_screen_size_based_height,
)


# This might seem like NIH, however the QtMessageBox is notoriously
# hard to work with if you need anything like for example resizing.
class ErtMessageBox(QDialog):
    def __init__(
        self,
        text: Optional[str],
        detailed_text: Optional[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.box = QDialogButtonBox(
            QDialogButtonBox.Ok,
        )
        self.box.setCenterButtons(True)
        self.box.accepted.connect(self.accept)

        self.details_text = QTextEdit()
        self.details_text.setPlainText(detailed_text)
        self.details_text.setObjectName("ErtMessageBox")
        self.details_text.setReadOnly(True)

        self.label_text = QLabel(text)
        self.label_icon = QLabel()
        style = self.style()
        assert style is not None
        icon = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical)
        self.label_icon.setPixmap(icon.pixmap(32))

        lay = QGridLayout(self)
        lay.addWidget(self.label_icon, 1, 1, 1, 1)
        lay.addWidget(self.label_text, 1, 2, 1, 2)
        lay.addWidget(self.details_text, 2, 1, 1, 3)
        lay.addWidget(self.box, 3, 1, 1, 3)

        fontWidth = self.box.fontMetrics().averageCharWidth()
        min_width = calculate_font_based_width(fontWidth)
        self.setMinimumSize(min_width, calculate_screen_size_based_height())
