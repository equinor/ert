from PyQt4 import QtGui, QtCore
import os
import time

def resourceIcon(name):
    """Load an image as an icon"""
    return QtGui.QIcon(os.path.dirname(__file__) + "/../../img/" + name)

def resourceStateIcon(on, off):
    """Load two images as an icon with on and off states"""
    icon = QtGui.QIcon()
    icon.addPixmap(resourceImage(on), state=QtGui.QIcon.On)
    icon.addPixmap(resourceImage(off), state=QtGui.QIcon.Off)
    return icon

def resourceImage(name):
    """Load an image as a Pixmap"""
    return QtGui.QPixmap(os.path.dirname(__file__) + "/../../img/" + name)


class ListCheckPanel(QtGui.QHBoxLayout):
    """
    Creates a panel with two buttons to select and unselect all elements of a list.
    A function: setSelectionEnabled(bool) is added to the list which enables enabling/disabling of list and buttons.
    """

    def __init__(self, list):
        QtGui.QHBoxLayout.__init__(self)

        list.checkAll = QtGui.QToolButton()
        list.checkAll.setIcon(resourceIcon("checked"))
        list.checkAll.setIconSize(QtCore.QSize(16, 16))
        list.checkAll.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        list.checkAll.setAutoRaise(True)
        list.checkAll.setToolTip("Select all")

        list.uncheckAll = QtGui.QToolButton()
        list.uncheckAll.setIcon(resourceIcon("notchecked"))
        list.uncheckAll.setIconSize(QtCore.QSize(16, 16))
        list.uncheckAll.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        list.uncheckAll.setAutoRaise(True)
        list.uncheckAll.setToolTip("Unselect all")

        self.setMargin(0)
        self.setSpacing(0)
        self.addStretch(1)
        self.addWidget(list.checkAll)
        self.addWidget(list.uncheckAll)

        self.connect(list.checkAll, QtCore.SIGNAL('clicked()'), list.selectAll)
        self.connect(list.uncheckAll, QtCore.SIGNAL('clicked()'), list.clearSelection)

        def setSelectionEnabled(bool):
            list.setEnabled(bool)
            list.checkAll.setEnabled(bool)
            list.uncheckAll.setEnabled(bool)

        list.setSelectionEnabled = setSelectionEnabled




class ValidatedTimestepCombo(QtGui.QComboBox):

    def __init__(self, parent, fromValue=0, fromLabel="Initial", toValue=1, toLabel="Final"):
        QtGui.QComboBox.__init__(self, parent)

        self.fromLabel = fromLabel
        self.fromValue = fromValue
        self.toLabel = toLabel
        self.toValue = toValue
        self.minTimeStep = fromValue
        self.maxTimeStep = toValue

        self.setMaximumWidth(150)
        self.setEditable(True)
        self.setValidator(QtGui.QIntValidator(self.minTimeStep, self.maxTimeStep, None))
        self.addItem(self.fromLabel + " (" + str(self.fromValue) + ")")
        self.addItem(self.toLabel + " (" + str(self.toValue) + ")")


    def focusOutEvent(self, event):
        QtGui.QComboBox.focusOutEvent(self, event)

        timestepMakesSense = False
        currentText = str(self.currentText())
        if currentText.startswith(self.fromLabel) or currentText.startswith(self.toLabel):
            timestepMakesSense = True

        elif currentText.isdigit():
            intValue = int(currentText)
            timestepMakesSense = True

            if intValue < self.minTimeStep:
                 self.setCurrentIndex(0)

            if intValue > self.maxTimeStep:
                 self.setCurrentIndex(1)


        if not timestepMakesSense:
            self.setCurrentIndex(0)


    def setMinTimeStep(self, value):
        self.minTimeStep = value
        self.validator().setBottom(value)

    def setMaxTimeStep(self, value):
        self.maxTimeStep = value
        self.validator().setTop(value)

    def setFromValue(self, value):
        self.fromValue = value
        self.setItemText(0, self.fromLabel + " (" + str(self.fromValue) + ")")

    def setToValue(self, value):
        self.toValue = value
        if self.toValue < self.fromValue:
            self.setItemText(1, self.toLabel)
        else:
            self.setItemText(1, self.toLabel + " (" + str(self.toValue) + ")")

    def setHistoryLength(self, length):
        self.setMaxTimeStep(length)
        self.setToValue(length)

    def getSelectedValue(self):
        currentText = str(self.currentText())

        if currentText.startswith(self.fromLabel):
            return self.fromValue
        elif currentText.startswith(self.toLabel):
            return self.toValue
        else:
            return int(currentText)



def createSeparator():
        """Creates a widget that can be used as a separator line on a panel."""
        qw = QtGui.QWidget()
        qwl = QtGui.QVBoxLayout()
        qw.setLayout(qwl)

        qf = QtGui.QFrame()
        qf.setFrameShape(QtGui.QFrame.HLine)
        qf.setFrameShadow(QtGui.QFrame.Sunken)

        qwl.addSpacing(5)
        qwl.addWidget(qf)
        qwl.addSpacing(5)

        return qw

def createSpace(size = 5):
    """Creates a widget that can be used as spacing on  a panel."""
    qw = QtGui.QWidget()
    qw.setMinimumSize(QtCore.QSize(size, size))

    return qw


def getItemsFromList(list, func = lambda item : str(item.text()), selected = True) :
    """Creates a list of strings from the selected items of a ListWidget or all items if selected is False"""
    if selected:
        selectedItemsList = list.selectedItems()
    else:
        selectedItemsList = []
        for index in range(list.count()):
            selectedItemsList.append(list.item(index))

    selectedItems = []
    for item in selectedItemsList:
        selectedItems.append(func(item))

    return selectedItems

    
def frange(*args):
    """
    A float range generator.
    Found here: http://code.activestate.com/recipes/66472/
    """
    start = 0.0
    step = 1.0

    l = len(args)
    if l == 1:
        end = args[0]
    elif l == 2:
        start, end = args
    elif l == 3:
        start, end, step = args
        if step == 0.0:
            raise ValueError, "step must not be zero"
    else:
        raise TypeError, "frange expects 1-3 arguments, got %d" % l

    v = start
    while True:
        if (step > 0 and v >= end) or (step < 0 and v <= end):
            raise StopIteration
        yield v
        v += step

def shortTime(secs):
    if secs == -1:
        return "-"
    else:
        t = time.localtime(secs)
        return time.strftime("%H:%M:%S", t)
