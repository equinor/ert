import os
import sys
from PyQt4.QtGui import QApplication
from ert_gui.pages.plot_panel import PlotPanel

def main():
    # QApplication.setGraphicsSystem("openvg")
    app = QApplication(sys.argv) #Early so that QT is initialized before other imports

    plot_panel = PlotPanel()
    plot_panel.show()

    sys.exit(app.exec_())



if __name__ == "__main__":
    main()







