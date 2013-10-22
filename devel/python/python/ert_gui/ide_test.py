import os
import sys
from PyQt4.QtGui import QApplication, QMainWindow, QVBoxLayout, QWidget
from ert_gui.ide.highlighter import KeywordHighlighter
from ert_gui.ide.ide_panel import IDEPanel
from ert_gui.widgets.search_box import SearchBox


def main():
    QApplication.setGraphicsSystem("raster")
    app = QApplication(sys.argv) #Early so that QT is initialized before other imports


    main_window = QMainWindow()

    central_widget = QWidget()
    main_window.setCentralWidget(central_widget)
    layout = QVBoxLayout()
    central_widget.setLayout(layout)

    search = SearchBox()


    ide = IDEPanel()

    layout.addWidget(search)
    layout.addWidget(ide, 1)

    os.chdir("/private/jpb/Projects/ert/TestCase")

    config_file = ""
    with open("config") as f:
        config_file = f.read()

    highlighter = KeywordHighlighter(ide.document())
    ide.handler_names = sorted(highlighter.handler_names)

    search.filterChanged.connect(highlighter.setSearchString)

    ide.document().setPlainText(config_file)

    cursor = ide.textCursor()
    cursor.setPosition(0)
    ide.setTextCursor(cursor)
    ide.setFocus()

    main_window.show()

    sys.exit(app.exec_())



if __name__ == "__main__":
    main()







