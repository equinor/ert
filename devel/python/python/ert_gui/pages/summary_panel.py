from PyQt4.QtGui import QFrame


class SummaryPanel(QFrame):
    def __init__(self, parent=None):
        QFrame.__init__(self, parent)

        self.setMinimumWidth(250)
        self.setMinimumHeight(250)
