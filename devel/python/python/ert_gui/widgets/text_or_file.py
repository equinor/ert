from ert_gui.widgets.option_widget import OptionWidget
from ert_gui.widgets.path_chooser import PathChooser

from ert_gui.widgets.string_box import StringBox

from ert_gui.models.mixins.connectorless import DefaultPathModel, StringModel
from ert_gui.ide.keywords.definitions import NumberListStringArgument


class TextOrFile(OptionWidget):
    """An option widget for text and file combo.

    The widget contains one tab with a text field, and one tab with a file
    chooser.

    """

    def __init__(self, setter, help_link=""):
        """
        Takes as argument a setter for the simulation model to set the current
        value of this widget.
        """
        OptionWidget.__init__(self, help_link=help_link)
        self.model_setter = setter

        iteration_weights_path_model = DefaultPathModel("", must_exist=True)
        iteration_weights_path_chooser = PathChooser(iteration_weights_path_model, path_label="Iteration weights file", help_link=help_link)
        iteration_weights_path_model.observable().attach(DefaultPathModel.PATH_CHANGED_EVENT, self._valueChanged)

        custom_iteration_weights_model = StringModel("1")
        custom_iteration_weights_box = StringBox(custom_iteration_weights_model, "Custom iteration weights", help_link=help_link, continuous_update=True)
        custom_iteration_weights_box.setValidator(NumberListStringArgument())
        custom_iteration_weights_model.observable().attach(StringModel.VALUE_CHANGED_EVENT, self._valueChanged)

        self.addHelpedWidget("Custom", custom_iteration_weights_box)
        self.addHelpedWidget("File", iteration_weights_path_chooser)

        # It is necessary to set a minimum height in some way;
        # otherwise the input field becomes invisible when the window
        # is resized to minimum vertical size. The value '50' is taken
        # out of thin air, but seems to work.
        self.setMinimumHeight(50)

    def isValid(self):
        """Returns the validation value"""
        value = self.getValue()
        return value is not None and not NumberListStringArgument().validate(value).failed()

    def getValue(self, joiner=','):
        """Get content of visible widget.  If current widget is the text field,
        we return the content of the text file and ignore parameter joiner.

        If current widget is the file path widget, we read the content of the
        file corresponding to the given filename, and join each line with
        joiner.
        """
        if self.getCurrentWidget() is None:
            return None

        if isinstance(self.getCurrentWidget(), StringBox):
            x = self.getCurrentWidget().model.getValue()
            return x

        fname = self.getCurrentWidget().model.getPath()
        if not fname:
            return ""
        result = self.parseFile(fname)
        return joiner.join(result)

    def setValue(self, value):
        self.getCurrentWidget().setValue(str(value))

    def _valueChanged(self):
        self.model_setter(self.getValue())

    def parseFile(self, fname):
        """Reads fname and returns a list of tokens.

        A token is a smallest contiguous string separated by a line break,
        space, tab, or semicolon.

        This parsing returns a list of tokens.
        """
        result = []
        with open(fname, 'r') as f:
            for line in f:
                line = line.replace(",", " ")
                line = line.replace(";", " ")
                d = line.strip().split()
                for x in d:
                    xstrip = x.strip()
                    if xstrip:
                        result.append(xstrip)
        return result
