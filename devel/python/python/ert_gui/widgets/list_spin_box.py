from PyQt4.QtGui import QSpinBox, QValidator


class ListSpinBox(QSpinBox):

    def __init__(self, items):
        QSpinBox.__init__(self)
        self.setMinimumWidth(75)

        self.__string_converter = str
        self.__items = items

        self.setRange(0, len(items) - 1)
        self.setValue(len(items) - 1)



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

        for index in range(len(self.__items)):
            value = self.convertToString(self.__items[index]).lower()

            if value == text:
                return QValidator.Acceptable, len(value)

            if value[0:pos] == text:
                return QValidator.Intermediate, pos

        return QValidator.Invalid, pos


    def fixup(self, input):
        text = str(input).lower()

        for index in range(len(self.__items)):
            value = self.convertToString(self.__items[index])

            if value.lower().startswith(text.lower()):
                input.clear()
                input.push_back(value)

