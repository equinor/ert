from PyQt4.QtCore import QSize, QRect, pyqtSignal, Qt
from PyQt4.QtGui import QWidget, QPainter, QHBoxLayout, QLabel, QComboBox, QVBoxLayout


class StyleChooser(QWidget):
    styleChanged = pyqtSignal(str, str)

    STYLE_OFF = ("Off", None)
    STYLE_AREA = ("Area", "#")
    STYLE_SOLID = ("Solid", "-")
    STYLE_DASHED = ("Dashed", "--")
    STYLE_DOTTED = ("Dotted", ":")
    STYLE_DASH_DOTTED = ("Dash Dotted", "-.")

    STYLES = [STYLE_OFF, STYLE_AREA, STYLE_SOLID, STYLE_DASHED, STYLE_DOTTED, STYLE_DASH_DOTTED]
    STYLES_LINE_ONLY = [STYLE_OFF, STYLE_SOLID, STYLE_DASHED, STYLE_DOTTED, STYLE_DASH_DOTTED]

    MARKER_OFF = ("Off", None)
    MARKER_X = ("X", "x")
    MARKER_CIRCLE = ("Circle", "o")
    MARKER_POINT = ("Point", ".")
    MARKER_STAR = ("Star", "*")
    MARKER_DIAMOND = ("Diamond", "D")

    MARKERS = [MARKER_OFF, MARKER_X, MARKER_CIRCLE, MARKER_POINT, MARKER_STAR, MARKER_DIAMOND]

    """Combines a StyleChooser with a label"""
    def __init__(self, label, line_style=STYLE_OFF, marker_style=MARKER_OFF, area_supported=False, labeled=False):
        QWidget.__init__(self)

        self.__styles = StyleChooser.STYLES if area_supported else StyleChooser.STYLES_LINE_ONLY

        self.setMinimumWidth(140)
        if labeled:
            self.setMaximumHeight(45)
        else:
            self.setMaximumHeight(25)

        layout = QHBoxLayout()
        layout.setMargin(0)
        layout.setSpacing(2)

        self.line_chooser = QComboBox()
        self.line_chooser.setToolTip("Select line style.")
        for style in self.__styles:
            self.line_chooser.addItem(*style)

        self.marker_chooser = QComboBox()
        self.marker_chooser.setToolTip("Select marker style.")
        for marker in StyleChooser.MARKERS:
            self.marker_chooser.addItem(*marker)

        self.style_label = QLabel("%s:" % label)
        self.style_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        layout.addWidget(self.style_label)

        if labeled:
            labeled_line_chooser = self._createLabeledChooser("Line Style", self.line_chooser)
            labeled_marker_chooser = self._createLabeledChooser("Marker", self.marker_chooser)
            layout.addWidget(labeled_line_chooser)
            layout.addWidget(labeled_marker_chooser)
        else:
            layout.addWidget(self.line_chooser)
            layout.addWidget(self.marker_chooser)

        self.setLayout(layout)
        self.label = label

        self.line_chooser.currentIndexChanged.connect(self.updateStyle)
        self.marker_chooser.currentIndexChanged.connect(self.updateStyle)

        self.line_chooser.setCurrentIndex(self.__styles.index(line_style))
        self.marker_chooser.setCurrentIndex(StyleChooser.MARKERS.index(marker_style))

    def _createLabeledChooser(self, label, chooser):
        labeled_line_chooser = QWidget()
        llc_layout = QVBoxLayout()
        llc_layout.setMargin(0)
        llc_layout.addWidget(QLabel(label))
        llc_layout.addWidget(chooser)
        labeled_line_chooser.setLayout(llc_layout)
        return labeled_line_chooser

    def updateLineStyleAndMarker(self, line_style, marker):
        self.line_chooser.setCurrentIndex(self.__styles.index(line_style))
        self.marker_chooser.setCurrentIndex(StyleChooser.MARKERS.index(marker))


    def updateStyle(self):
        self.marker_chooser.setEnabled(self.line_chooser.currentText() != "Area")

        line_style = self.line_chooser.itemData(self.line_chooser.currentIndex())
        marker_style = self.marker_chooser.itemData(self.marker_chooser.currentIndex())

        line_style = str(line_style.toString())
        marker_style = str(marker_style.toString())

        self.styleChanged.emit(line_style, marker_style)