#  Copyright (C) 2014  Statoil ASA, Norway.
#
#  The file 'closable_dialog.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.

from PyQt4.QtCore import Qt, QSize
from PyQt4.QtGui import  QDialog, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFont
from ert.util import Version
from ert_gui.widgets import util


class AboutDialog(QDialog):

    def __init__(self, parent):
        QDialog.__init__(self, parent)

        self.setWindowTitle("Version")
        self.setModal(True)
        self.setFixedSize(QSize(400, 240))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCancelButtonHint)

        layout = QVBoxLayout()

        image = util.resourceImage("splash.jpg")

        scene = QGraphicsScene(self)
        view = QGraphicsView(scene)
        image = image.scaledToHeight(160,Qt.SmoothTransformation)
        item = QGraphicsPixmapItem(image)
        scene.addItem(item)
        scene.setSceneRect(0,0,image.width(),image.height())
        view.setFixedSize(image.width(),image.height())
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.show()

        inner_layout = QHBoxLayout()
        inner_layout.addWidget(view)

        layout.addLayout(inner_layout)
        text_layout = QVBoxLayout()

        ert = QLabel()
        title_font = QFont()
        title_font.setPointSize(40)
        ert.setFont(title_font)
        ert.setText("ERT")
        title_layout = QHBoxLayout()
        title_layout.addStretch()
        title_layout.addWidget(ert)
        title_layout.addStretch()

        text_layout.addLayout(title_layout)

        ert_title = QLabel()
        ert_title.setText("Ensemble based Reservoir Tool")

        text_layout.addWidget(ert_title)

        version = QLabel()
        version.setText("Version: %s" % Version.getVersion())
        text_layout.addWidget(version)

        timestamp = QLabel()
        timestamp.setText("Build time: %s" % Version.getBuildTime())
        text_layout.addWidget(timestamp)
        text_layout.addStretch()

        inner_layout.addLayout(text_layout)

        self.__button_layout = QHBoxLayout()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.__button_layout.addStretch()
        self.__button_layout.addWidget(self.close_button)
        self.__button_layout.addStretch()
        layout.addLayout(self.__button_layout)

        self.setLayout(layout)


    def keyPressEvent(self, q_key_event):
        if not self.close_button.isEnabled() and q_key_event.key() == Qt.Key_Escape:
            pass
        else:
            QDialog.keyPressEvent(self, q_key_event)


