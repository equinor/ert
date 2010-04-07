from PyQt4 import QtGui, QtCore
from helpedwidget import *

class AddRemoveWidget(QtGui.QWidget):
    """
    A simple class that provides to vertically positioned buttons for adding and removing something.
    The addFunction and removeFunction functions must be provided.
    """
    def __init__(self, parent=None, addFunction=None, removeFunction=None):
        """Creates a two button widget"""
        QtGui.QWidget.__init__(self, parent)

        addButton = QtGui.QToolButton(self)
        addButton.setIcon(QtGui.QIcon.fromTheme("add"))
        addButton.setIconSize(QtCore.QSize(16, 16))
        self.connect(addButton, QtCore.SIGNAL('clicked()'), addFunction)

        removeButton = QtGui.QToolButton(self)
        removeButton.setIcon(QtGui.QIcon.fromTheme("remove"))
        removeButton.setIconSize(QtCore.QSize(16, 16))
        self.connect(removeButton, QtCore.SIGNAL('clicked()'), removeFunction)

        buttonLayout = QtGui.QVBoxLayout()
        buttonLayout.setMargin(0)
        buttonLayout.addWidget(addButton)
        buttonLayout.addWidget(removeButton)
        buttonLayout.addStretch(1)
        self.setLayout(buttonLayout)



class KeywordList(HelpedWidget):
    """Shows a list of keywords. The data structure expected and sent to the getter and setter is an array of values."""
    def __init__(self, parent=None, listLabel="", help=""):
        """Construct a list for showing keywords"""
        HelpedWidget.__init__(self, parent, listLabel, help)

        self.list = QtGui.QListWidget(self)
        self.list.setMinimumHeight(50)
        self.list.setMaximumHeight(70)

        self.addWidget(self.list)

        self.addWidget(AddRemoveWidget(self, self.addItem, self.removeItem))

        self.addStretch()
        self.addHelpButton()


    def addItem(self):
        """Called by the add button to insert a new keyword"""
        newKeyWord, ok = QtGui.QInputDialog.getText(self, self.tr("QInputDialog.getText()"), self.tr("Keyword:"), QtGui.QLineEdit.Normal )
        newKeyWord = str(newKeyWord).strip()

        if ok and not newKeyWord == "":
            self.list.addItem(newKeyWord)
            self.contentsChanged()


    def removeItem(self):
        """Called by the remove button to remove a selected keyword"""
        if not self.list.currentItem() == None:
            self.list.takeItem(self.list.currentRow())
            self.contentsChanged()


    def contentsChanged(self):
        """Called whenever the contents of the list changes."""
        keywordList = []
        for index in range(self.list.count()):
            keywordList.append(str(self.list.item(index).text()))

        self.updateContent(keywordList)


    def fetchContent(self):
        """Retrieves data from the model and inserts it into the list"""
        keywords = self.getFromModel()

        self.list.clear()

        for keyword in keywords:
            self.list.addItem(keyword)




class KeywordTable(HelpedWidget):
    """Shows a table of key/value pairs. The data structure expected and sent to the getter and setter is a dictionary of values."""
    def __init__(self, parent=None, tableLabel="", help="", colHead1="Keyword", colHead2="Value"):
        """Construct a table for key/value pairs."""
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

        self.addWidget(AddRemoveWidget(self, self.addItem, self.removeItem))

        self.addHelpButton()

        #self.connect(self.spinner, QtCore.SIGNAL('valueChanged(int)'), self.updateContent)
        self.connect(self.table, QtCore.SIGNAL('cellChanged(int,int)'), self.contentsChanged)


    def addItem(self):
        """Called by the add button to insert a new keyword"""
        self.table.insertRow(self.table.currentRow() + 1)
        self.contentsChanged()


    def removeItem(self):
        """Called by the remove button to remove a selected keyword"""
        currentRow = self.table.currentRow()

        if currentRow >= 0:
            doDelete = QtGui.QMessageBox.question(self, "Delete row?", "Are you sure you want to delete the key/value pair?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No )

            if doDelete:
                self.table.removeRow(currentRow)
                self.contentsChanged()


    def contentsChanged(self):
        """Called whenever the contents of a cell changes."""
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
        """Retrieves data from the model and inserts it into the table."""
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
    """Shows a table of parameters. The data structure expected and sent to the getter and setter is an array of arrays."""
    def __init__(self, parent=None, tableLabel="", help="", colHeads=["c1", "c2", "c3", "c4", "c5"]):
        """Construct a table with arbitrary number of columns."""
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


        self.addWidget(AddRemoveWidget(self, self.addItem, self.removeItem))

        self.addHelpButton()

        #self.connect(self.spinner, QtCore.SIGNAL('valueChanged(int)'), self.updateContent)
        self.connect(self.table, QtCore.SIGNAL('cellChanged(int,int)'), self.contentsChanged)


    def addItem(self):
        """Called by the add button to insert a new keyword"""
        self.table.insertRow(self.table.currentRow() + 1)
        self.contentsChanged()


    def removeItem(self):
        """Called by the remove button to remove a selected keyword"""
        currentRow = self.table.currentRow()

        if currentRow >= 0:
            doDelete = QtGui.QMessageBox.question(self, "Delete row?", "Are you sure you want to delete the key/value pair?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No )

            if doDelete:
                self.table.removeRow(currentRow)
                self.contentsChanged()


    def contentsChanged(self):
        """Called whenever the contents of a cell changes."""
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
        """Retrieves data from the model and inserts it into the table"""
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

    def setDelegate(self, column, delegate):
        self.table.setItemDelegateForColumn(column, delegate)



class SpinBoxDelegate(QtGui.QItemDelegate):
    def __init__(self, parent):
        QtGui.QItemDelegate.__init__(self, parent)

    # QWidget *parent, const QStyleOptionViewItem &/* option */, const QModelIndex &/* index */
    def createEditor(self, parent, option, index):
        editor = QtGui.QSpinBox(parent)

        editor.setMinimum(0)
        editor.setMaximum(100)

        return editor

    #QWidget *editor, const QModelIndex &index
    def setEditorData(self, editor, index):
        value = index.model().data(index, QtCore.Qt.EditRole).toInt()
        editor.setValue(value[0])

    #QWidget *editor, QAbstractItemModel *model, const QModelIndex &index
    def setModelData(self, editor, model, index):
        editor.interpretText()
        value = editor.value()

        model.setData(index, value, QtCore.Qt.EditRole)

    #Widget *editor, const QStyleOptionViewItem &option, const QModelIndex &/* index */
    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class DoubleSpinBoxDelegate(QtGui.QItemDelegate):
    def __init__(self, parent):
        QtGui.QItemDelegate.__init__(self, parent)

    # QWidget *parent, const QStyleOptionViewItem &/* option */, const QModelIndex &/* index */
    def createEditor(self, parent, option, index):
        editor = QtGui.QDoubleSpinBox(parent)

        editor.setMinimum(0)
        editor.setMaximum(100)
        editor.setDecimals(2)
        editor.setSingleStep(0.01)

        return editor

    #QWidget *editor, const QModelIndex &index
    def setEditorData(self, editor, index):
        value = index.model().data(index, QtCore.Qt.EditRole).toDouble()
        editor.setValue(value[0])

    #QWidget *editor, QAbstractItemModel *model, const QModelIndex &index
    def setModelData(self, editor, model, index):
        editor.interpretText()
        value = editor.value()

        model.setData(index, value, QtCore.Qt.EditRole)

    #Widget *editor, const QStyleOptionViewItem &option, const QModelIndex &/* index */
    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class ComboBoxDelegate(QtGui.QItemDelegate):
    def __init__(self, parent):
        QtGui.QItemDelegate.__init__(self, parent)

    # QWidget *parent, const QStyleOptionViewItem &/* option */, const QModelIndex &/* index */
    def createEditor(self, parent, option, index):
        editor = QtGui.QComboBox(parent)

        editor.setMinimum(0)
        editor.setMaximum(100)
        editor.setDecimals(2)
        editor.setSingleStep(0.01)

        return editor

    #QWidget *editor, const QModelIndex &index
    def setEditorData(self, editor, index):
        value = index.model().data(index, QtCore.Qt.EditRole).toDouble()
        editor.setValue(value[0])

    #QWidget *editor, QAbstractItemModel *model, const QModelIndex &index
    def setModelData(self, editor, model, index):
        editor.interpretText()
        value = editor.value()

        model.setData(index, value, QtCore.Qt.EditRole)

    #Widget *editor, const QStyleOptionViewItem &option, const QModelIndex &/* index */
    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

