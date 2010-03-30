from PyQt4 import QtGui, QtCore
from helpedwidget import *

class KeywordList(HelpedWidget):
    def __init__(self, parent=None, listLabel="", help=""):
        """Construct a spinner widget for integers"""
        HelpedWidget.__init__(self, parent, listLabel, help)

        self.list = QtGui.QListWidget(self)
        self.list.setMinimumHeight(50)
        self.list.setMaximumHeight(70)

        self.addWidget(self.list)

        buttonWidget = QtGui.QWidget(self)
        addButton = QtGui.QToolButton(self)
        addButton.setIcon(QtGui.QIcon.fromTheme("add"))
        addButton.setIconSize(QtCore.QSize(16, 16))
        self.connect(addButton, QtCore.SIGNAL('clicked()'), self.addItem)

        removeButton = QtGui.QToolButton(self)
        removeButton.setIcon(QtGui.QIcon.fromTheme("remove"))
        removeButton.setIconSize(QtCore.QSize(16, 16))
        self.connect(removeButton, QtCore.SIGNAL('clicked()'), self.removeItem)

        buttonLayout = QtGui.QVBoxLayout()
        buttonLayout.setMargin(0)
        buttonLayout.addWidget(addButton)
        buttonLayout.addWidget(removeButton)
        buttonLayout.addStretch(1)
        buttonWidget.setLayout(buttonLayout)
        self.addWidget(buttonWidget)

        self.addStretch()
        self.addHelpButton()

        #self.connect(self.spinner, QtCore.SIGNAL('valueChanged(int)'), self.updateContent)
        #self.connect(self.spinner, QtCore.SIGNAL('editingFinished()'), self.contentsChanged)

    def addItem(self):
        newKeyWord, ok = QtGui.QInputDialog.getText(self, self.tr("QInputDialog.getText()"), self.tr("Keyword:"), QtGui.QLineEdit.Normal )
        newKeyWord = str(newKeyWord).strip()

        if ok and not newKeyWord == "":
            self.list.addItem(newKeyWord)
            self.contentsChanged()

    def removeItem(self):
        if not self.list.currentItem() == None:
            self.list.takeItem(self.list.currentRow())
            self.contentsChanged()


    def contentsChanged(self):

        keywordList = []
        for index in range(self.list.count()):
            keywordList.append(self.list.item(index).text())

        self.updateContent(keywordList)

    def fetchContent(self):
        keywords = self.getFromModel()

        self.list.clear()

        for keyword in keywords:
            self.list.addItem(keyword)


class KeywordTable(HelpedWidget):
    def __init__(self, parent=None, tableLabel="", help="", colHead1="Keyword", colHead2="Value"):
        """Construct a spinner widget for integers"""
        HelpedWidget.__init__(self, parent, tableLabel, help)

        self.table = QtGui.QTableWidget(self)
        self.table.setColumnCount(2)
        self.headers = [colHead1, colHead2]
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.setColumnWidth(0, 150)
        #self.table.setColumnWidth(1, 250)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.table.setMinimumHeight(110)
        self.table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

        self.addWidget(self.table)

        buttonWidget = QtGui.QWidget(self)
        addButton = QtGui.QToolButton(self)
        addButton.setIcon(QtGui.QIcon.fromTheme("add"))
        addButton.setIconSize(QtCore.QSize(16, 16))
        self.connect(addButton, QtCore.SIGNAL('clicked()'), self.addItem)

        removeButton = QtGui.QToolButton(self)
        removeButton.setIcon(QtGui.QIcon.fromTheme("remove"))
        removeButton.setIconSize(QtCore.QSize(16, 16))
        self.connect(removeButton, QtCore.SIGNAL('clicked()'), self.removeItem)

        buttonLayout = QtGui.QVBoxLayout()
        buttonLayout.setMargin(0)
        buttonLayout.addWidget(addButton)
        buttonLayout.addWidget(removeButton)
        buttonLayout.addStretch(1)
        buttonWidget.setLayout(buttonLayout)
        self.addWidget(buttonWidget)

        self.addHelpButton()

        #self.connect(self.spinner, QtCore.SIGNAL('valueChanged(int)'), self.updateContent)
        self.connect(self.table, QtCore.SIGNAL('cellChanged(int,int)'), self.contentsChanged)


    def addItem(self):
        self.table.insertRow(self.table.currentRow() + 1)
        self.contentsChanged()


    def removeItem(self):
        currentRow = self.table.currentRow()

        if currentRow >= 0:
            doDelete = QtGui.QMessageBox.question(self, "Delete row?", "Are you sure you want to delete the key/value pair?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No )

            if doDelete:
                self.table.removeRow(currentRow)
                self.contentsChanged()


    def contentsChanged(self):
        keyValueList = {}

        for index in range(self.table.rowCount()):
            key = self.table.item(index, 0)
            if not key == None:
                key = str(key.text()).strip()
                value = self.table.item(index, 1)

                if not key == "" and not value == None:
                    keyValueList[key] = str(value.text()).strip()

        self.updateContent(keyValueList)


    def fetchContent(self):
        keywords = self.getFromModel()

        self.table.clear()
        self.table.setHorizontalHeaderLabels(self.headers)

        row = 0
        for key in keywords.keys():
            keyItem = QtGui.QTableWidgetItem(key)
            valueItem = QtGui.QTableWidgetItem(keywords[key])
            self.table.insertRow(row)
            self.table.setItem(row, 0, keyItem)
            self.table.setItem(row, 1, valueItem)
            row+=1




class MultiColumnTable(HelpedWidget):
    def __init__(self, parent=None, tableLabel="", help="", colHeads=["c1", "c2", "c3", "c4", "c5"]):
        """Construct a spinner widget for integers"""
        HelpedWidget.__init__(self, parent, tableLabel, help)

        self.table = QtGui.QTableWidget(self)
        self.table.setColumnCount(len(colHeads))
        self.headers = colHeads
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.setColumnWidth(0, 150)
        #self.table.setColumnWidth(1, 250)
        #self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        #self.table.horizontalHeader().setResizeMode(QtGui.QHeaderView.Interactive)

        self.table.setMinimumHeight(110)
        self.table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

        self.addWidget(self.table)

        buttonWidget = QtGui.QWidget(self)
        addButton = QtGui.QToolButton(self)
        addButton.setIcon(QtGui.QIcon.fromTheme("add"))
        addButton.setIconSize(QtCore.QSize(16, 16))
        self.connect(addButton, QtCore.SIGNAL('clicked()'), self.addItem)

        removeButton = QtGui.QToolButton(self)
        removeButton.setIcon(QtGui.QIcon.fromTheme("remove"))
        removeButton.setIconSize(QtCore.QSize(16, 16))
        self.connect(removeButton, QtCore.SIGNAL('clicked()'), self.removeItem)

        buttonLayout = QtGui.QVBoxLayout()
        buttonLayout.setMargin(0)
        buttonLayout.addWidget(addButton)
        buttonLayout.addWidget(removeButton)
        buttonLayout.addStretch(1)
        buttonWidget.setLayout(buttonLayout)
        self.addWidget(buttonWidget)

        self.addHelpButton()

        #self.connect(self.spinner, QtCore.SIGNAL('valueChanged(int)'), self.updateContent)
        self.connect(self.table, QtCore.SIGNAL('cellChanged(int,int)'), self.contentsChanged)


    def addItem(self):
        self.table.insertRow(self.table.currentRow() + 1)
        self.contentsChanged()


    def removeItem(self):
        currentRow = self.table.currentRow()

        if currentRow >= 0:
            doDelete = QtGui.QMessageBox.question(self, "Delete row?", "Are you sure you want to delete the key/value pair?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No )

            if doDelete:
                self.table.removeRow(currentRow)
                self.contentsChanged()


    def contentsChanged(self):
        rowValues = []

        for rowIndex in range(self.table.rowCount()):
            row = []
            for columnIndex in range(self.table.columnCount()):
                item = self.table.item(rowIndex, columnIndex)
                if not item == None:
                    row.append(str(item.text()))
                else:
                    row.append("")

            rowValues.append(row)


        self.updateContent(rowValues)


    def fetchContent(self):
        rows = self.getFromModel()

        self.table.clear()
        self.table.setHorizontalHeaderLabels(self.headers)

        rowIndex = 0
        for row in rows:
            self.table.insertRow(rowIndex)
            columnIndex = 0
            for value in row:
                item = QtGui.QTableWidgetItem(str(value))                
                self.table.setItem(rowIndex, columnIndex, item)
                columnIndex+=1

            rowIndex+=1



