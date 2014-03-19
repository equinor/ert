import re
from PyQt4 import QtGui
from PyQt4.QtCore import QSize, QString, QStringList, Qt
from PyQt4.QtGui import QSpinBox, QValidator, QLineEdit, QCompleter, QListView, QStringListModel, QScrollBar
from ert.util import ctime


class ListSpinBox(QSpinBox):

    def __init__(self, items):
        QSpinBox.__init__(self)
        self.setMinimumWidth(75)

        self.__string_converter = str
        self.__items = items
        list = QStringList()
        for i,item in enumerate(self.__items):
            assert isinstance(item, ctime)
            date = item.date()
            list.append(self.convertToString(date))

        model = QStringListModel()
        model.setStringList(list)


        self.setRange(0, len(items) - 1)
        self.setValue(len(items) - 1)
        line_edit = QLineEdit()
        self.__completer = QCompleter()
        self.__completer.setModel(model)
        self.__completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)

        view = QListView()
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.__completer.setPopup(view)
        view.setAlternatingRowColors(True)
        view.setMinimumWidth(90)
        line_edit.setCompleter(self.__completer)
        line_edit.textEdited.connect(self.showCompleter)
        self.setLineEdit(line_edit)

    def showCompleter(self):
        self.__completer.complete()

    def convertToString(self, item):
        return self.__string_converter(item)

    def setStringConverter(self, string_converter):
        self.__string_converter = string_converter


    def textFromValue(self, value):
        if len(self.__items) == 0:
            return ""

        if value < 0 or value >= len(self.__items):
            value = len(self.__items) - 1

        return self.convertToString(self.__items[value])


    def valueFromText(self, qstring):
        text = str(qstring).lower()

        for index in range(len(self.__items)):
            value = self.convertToString(self.__items[index]).lower()
            if text == value:
                return index

        return len(self.__items) - 1


    def validate(self, qstring, pos):
        text = str(qstring).lower()

        if re.match("^[0-9-]+$", text) is None:
            return QValidator.Invalid, pos

        if len(text) < 10:
            return QValidator.Acceptable, pos

        for index in range(len(self.__items)):
            value = self.convertToString(self.__items[index]).lower()

            if value == text:
                return QValidator.Acceptable, len(value)

            if value[0:pos] == text:
                return QValidator.Intermediate, pos

        return QValidator.Invalid, pos
        #return QValidator.Acceptable, pos


    def fixup(self, input):
        text = str(input).lower()

        for index in range(len(self.__items)):
            value = self.convertToString(self.__items[index])

            if value.lower().startswith(text.lower()):
                input.clear()
                input.push_back(value)

