from qtpy.QtGui import QFont
from qtpy.QtWidgets import QLabel


class ActiveLabel(QLabel):
    def __init__(self, model):
        QLabel.__init__(self)

        self._model = model

        font = self.font()
        font.setWeight(QFont.Bold)
        self.setFont(font)

        self._model.valueChanged.connect(self.updateLabel)

        self.updateLabel()

    def updateLabel(self):
        """Retrieves data from the model and inserts it into the edit line"""
        model_value = self._model.getValue()
        if model_value is None:
            model_value = ""

        self.setText(str(model_value))
