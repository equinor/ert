from ert_gui.models import ErtConnector
from ert_gui.models.mixins import BasicModelMixin
from ert_gui.widgets.option_widget import OptionWidget
from ert_gui.models.mixins.connectorless import DefaultPathModel
from ert_gui.models.mixins.connectorless import StringModel
from ert_gui.widgets.path_chooser import PathChooser


from ert_gui.widgets.string_box import StringBox

from ert_gui.models.mixins.connectorless import DefaultPathModel, DefaultNameFormatModel, StringModel
from ert_gui.ide.keywords.definitions import ProperNameFormatArgument, NumberListStringArgument


class TextOrFileModel(ErtConnector, BasicModelMixin):

    optionWidget = OptionWidget()
    def __init__(self):
        super(TextOrFileModel, self).__init__()
        iteration_weights_path_model = DefaultPathModel("", must_exist=True)
        iteration_weights_path_chooser = PathChooser(iteration_weights_path_model, path_label="Iteration weights file")

        custom_iteration_weights_model = StringModel("1")
        custom_iteration_weights_box = StringBox(custom_iteration_weights_model, "Custom iteration weights", "config/simulation/iteration_weights")
        custom_iteration_weights_box.setValidator(NumberListStringArgument())
        self.optionWidget.addHelpedWidget("Custom", custom_iteration_weights_box)
        self.optionWidget.addHelpedWidget("File", iteration_weights_path_chooser)



    def getValue(self, joiner = ','):
        """Get content of visible widget.  If current widget is the text field,
        we return the content of the text file and ignore parameter joiner.

        If current widget is the file path widget, we read the content of the
        file corresponding to the given filename, and join each line with
        joiner.
        """
        if self.optionWidget.getCurrentWidget() == 0:
            return self.optionWidget.getCurrentWidget().getValue()
        fname = self.optionWidget.getCurrentWidget().getValue()
        result = []
        with open(fname, 'r') as f:
            for line in f:
                result.append(line)
        return joiner.join(result)

    def setValue(self, value):
        self.optionWidget.getCurrentWidget().setValue(value)
