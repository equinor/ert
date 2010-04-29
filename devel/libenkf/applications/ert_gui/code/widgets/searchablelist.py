from widgets.tablewidgets import AddRemoveWidget
from PyQt4 import QtGui, QtCore

class SearchableList(QtGui.QWidget):
    """
    A searchable list of items.
    Emits addItem(QListWidget) and removeItem(QListWidget) when the add and remove buttons are pressed.
    """
    passiveColor = QtGui.QColor(194, 194, 194)

    def __init__(self, parent=None, converter=lambda item : str(item.text()), list_height=350, list_width = 130, ignore_case=False):
        QtGui.QWidget.__init__(self, parent)
        self.setMaximumWidth(list_width)
        self.setMinimumWidth(list_width)
        self.converter = converter
        self.ignore_case = ignore_case

        vlayout = QtGui.QVBoxLayout()
        vlayout.setMargin(0)

        self.searchBox = QtGui.QLineEdit(parent)
        self.searchBox.setToolTip("Type to search!")
        self.searchBox.focusInEvent = lambda event : self.enterSearch(event)
        self.searchBox.focusOutEvent = lambda event : self.exitSearch(event)
        self.activeColor = self.searchBox.palette().color(self.searchBox.foregroundRole())
        self.disableSearch = True
        self.presentSearch()
        self.connect(self.searchBox, QtCore.SIGNAL('textChanged(QString)'), self.searchInList)
        vlayout.addWidget(self.searchBox)

        self.list = QtGui.QListWidget(parent)
        self.list.setMaximumWidth(list_width - 2)
        self.list.setMinimumWidth(list_width - 2)
        self.list.setMinimumHeight(list_height)
        self.list.setSortingEnabled(True)
        vlayout.addWidget(self.list)
        addItem = lambda : self.emit(QtCore.SIGNAL("addItem(list)"), self.list)
        removeItem = lambda : self.emit(QtCore.SIGNAL("removeItem(list)"), self.list)
        vlayout.addWidget(AddRemoveWidget(self, addItem, removeItem, True))
        self.setLayout(vlayout)

        def emitter(current, previous):
            self.emit(QtCore.SIGNAL("currentItemChanged(QListWidgetItem, QListWidgetItem)"), current, previous)

        self.connect(self.list, QtCore.SIGNAL('currentItemChanged(QListWidgetItem *, QListWidgetItem *)'), emitter)

    def presentSearch(self):
        """Is called to present the greyed out search"""
        self.disableSearch = True
        self.searchBox.setText("Search")
        palette = self.searchBox.palette()
        palette.setColor(self.searchBox.foregroundRole(), self.passiveColor)
        self.searchBox.setPalette(palette)

    def activateSearch(self):
        """Is called to remove the greyed out search"""
        self.disableSearch = False
        self.searchBox.setText("")
        palette = self.searchBox.palette()
        palette.setColor(self.searchBox.foregroundRole(), self.activeColor)
        self.searchBox.setPalette(palette)

    def enterSearch(self, focusEvent):
        """Called when the line edit gets the focus"""
        QtGui.QLineEdit.focusInEvent(self.searchBox, focusEvent)
        if str(self.searchBox.text()) == "Search":
            self.activateSearch()

    def exitSearch(self, focusEvent):
        """Called when the line edit looses focus"""
        QtGui.QLineEdit.focusOutEvent(self.searchBox, focusEvent)
        if str(self.searchBox.text()) == "":
            self.presentSearch()

    def searchInList(self, value):
        """Called when the contents of the search box changes"""
        if not self.disableSearch:
            for index in range(self.list.count()):
                item = self.list.item(index)
                text = self.converter(item)

                if self.ignore_case:
                    value = str(value).lower()
                    text = text.lower()

                if not text.find(value) == -1:
                    item.setHidden(False)
                else:
                    item.setHidden(True)

    def getList(self):
        return self.list