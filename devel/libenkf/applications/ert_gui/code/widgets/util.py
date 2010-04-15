from PyQt4 import QtGui, QtCore
import os

def resourceIcon(name):
    return QtGui.QIcon(os.path.dirname(__file__) + "/../../img/" + name)

def resourceImage(name):
    return QtGui.QPixmap(os.path.dirname(__file__) + "/../../img/" + name)