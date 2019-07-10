try:
    from PyQt5 import *
    QT4 = False
    QT5 = True
except ImportError:
    from PyQt4 import *
    QT4 = True
    QT5 = False

