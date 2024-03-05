from qtpy.QtCore import Slot
from qtpy.QtWidgets import (
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ert.storage import Ensemble, Experiment


class StorageInfoWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self._header = QLabel()
        # Magic number (sizeHint of QLineEdit) inorder to align better with StorageWidget
        self._header.setMaximumHeight(23)
        self._header.setMinimumHeight(23)

        self._info_area = QTextEdit()
        self._info_area.setObjectName("html_text")
        self._info_area.setReadOnly(True)
        self._info_area.setMinimumHeight(300)

        layout = QVBoxLayout()
        layout.addWidget(self._header)
        layout.addWidget(self._info_area)
        self.setLayout(layout)

    @Slot(Ensemble)
    def setEnsemble(self, ensemble: Ensemble):
        self._header.setText("Ensemble info")
        self._info_area.clear()
        html = "<table>"
        for state_index, value in enumerate(ensemble.get_ensemble_state()):
            html += f"<tr><td width=30>{state_index:d}.</td><td>{value.name}</td></tr>"
        html += "</table>"
        self._info_area.setHtml(html)

    @Slot(Experiment)
    def setExperiment(self, experiment: Experiment):
        self._header.setText("Experiment info")
        self._info_area.clear()
